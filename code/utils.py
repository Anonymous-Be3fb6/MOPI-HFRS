import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.nn.functional import cosine_similarity
import heapq
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch.nn.functional as F
import re
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, BertForSequenceClassification
import random


def set_seed(seed):
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # CUDA
    random.seed(seed)


"""
For Language Models
"""


def label_conversion(tokenizer, labels):
    yes_token_id = tokenizer.convert_tokens_to_ids('yes')
    no_token_id = tokenizer.convert_tokens_to_ids('no')

    # Assuming 'binary_labels' is your array of 0s and 1s
    modified_labels = [yes_token_id if label == 1 else no_token_id for label in labels]
    labels = torch.tensor(modified_labels)
    return labels


def train_reasoning(model, tokenizer, train_loader, optimizer, scheduler, criterion, accelerator):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(train_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = label_conversion(tokenizer, batch['labels'])

        optimizer.zero_grad()
        with accelerator.autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            next_word_logits = logits[:, -1, :]

            print(next_word_logits)
            print(next_word_logits.device)
            print(labels.device)

            loss = criterion(next_word_logits, labels)
            total_loss += loss

            preds = torch.argmax(next_word_logits, dim=1)
            correct_predictions += (preds == labels).sum().item()

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader.dataset)
    return avg_loss, accuracy


def evaluate_reasoning(model, tokenizer, data_loader, criterion, accelerator):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = label_conversion(tokenizer, batch['labels'])
            labels = labels

            with accelerator.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                next_word_logits = logits[:, -1, :]

                loss = criterion(next_word_logits, labels)
                total_loss += loss

                preds = torch.argmax(next_word_logits, dim=1)
                correct_predictions += (preds == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy


def train(model, train_loader, optimizer, scheduler, accelerator):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(train_loader):
        # batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        with accelerator.autocast():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (preds == batch['labels']).sum().item()

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader.dataset)
    return avg_loss, accuracy


def evaluate(model, data_loader, accelerator):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            # batch = {k: v.to(device) for k, v in batch.items()}

            with accelerator.autocast():
                outputs = model(**batch)

                loss = outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (preds == batch['labels']).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def extract_food_description(text):
    pattern = r"The food description is: (.*?), the nutrition component vector is"
    match = re.search(pattern, text)
    return match.group(1)


def tokenize(texts, max_length=256, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
    return tokenizer(texts, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')


def extract_descriptions(graph, idx_list, prompt_type='food'):
    descriptions = []
    for idx in idx_list:
        prompt = graph[prompt_type].prompt[idx]
        if prompt_type == 'food':
            description = extract_food_description(prompt)
        elif prompt_type == 'habit':
            description = prompt.split('The habit description is: ')[1]
        elif prompt_type == 'ingredient':
            description = prompt.split('The ingredient description is: ')[1]
        else:
            raise ValueError('Invalid prompt type')
        descriptions.append(description)
    return descriptions


def generate_prompted_text(graph, edge_list):
    user2food = generate_edge_mapping(graph, ('user', 'eats', 'food'))
    user2habit = generate_edge_mapping(graph, ('user', 'has', 'habit'))
    prompted_text_list = []
    for src, tgt in tqdm(edge_list.T):
        src, tgt = src.item(), tgt.item()
        src_user_description = graph['user'].prompt[src]
        tgt_user_description = graph['user'].prompt[tgt]
        # Extract shared food information
        src_foods, tgt_foods = user2food[src], user2food[tgt]
        unique_src_foods = list(set(src_foods) - set(tgt_foods))
        unique_tgt_foods = list(set(tgt_foods) - set(src_foods))
        src_food_descriptions = extract_descriptions(graph, unique_src_foods, prompt_type='food')
        tgt_food_descriptions = extract_descriptions(graph, unique_tgt_foods, prompt_type='food')
        shared_foods = list(set(src_foods).intersection(set(tgt_foods)))
        food_descriptions = extract_descriptions(graph, shared_foods, prompt_type='food')

        # Extract shared habit information
        src_habits, tgt_habits = user2habit[src], user2habit[tgt]
        shared_habits = list(set(src_habits).intersection(set(tgt_habits)))
        habit_descriptions = extract_descriptions(graph, shared_habits, prompt_type='habit')
        unique_src_habits = list(set(src_habits) - set(tgt_habits))
        unique_tgt_habits = list(set(tgt_habits) - set(src_habits))
        src_habit_descriptions = extract_descriptions(graph, unique_src_habits, prompt_type='habit')
        tgt_habit_descriptions = extract_descriptions(graph, unique_tgt_habits, prompt_type='habit')

        prompted_text = f"Act as a nutritionist, your task is to use the following food and habits to decide if the two users share a similar lifestyle:" \
                        f"Important Note: Your output should be limited to a single yes or no and provide a short sentence of explanation. " \
                        f"The first use is {src_user_description}. The second user is {tgt_user_description}. " \
                        f"Their shared foods are {'; '.join(food_descriptions)}. " \
                        f"Their shared habits are {'; '.join(habit_descriptions)}." \
                        f"The first user's unique foods are {'; '.join(src_food_descriptions)}. " \
                        f"The second user's unique foods are {'; '.join(tgt_food_descriptions)}. " \
                        f"The first user's unique habits are {'; '.join(src_habit_descriptions)}. " \
                        f"The second user's unique habits are {'; '.join(tgt_habit_descriptions)}. "
        prompted_text_list.append(prompted_text)

    return prompted_text_list


"""
Functions for data processing
"""


def metapath_generation(graph, sample_ratio=0.3, seed=42):
    meta_paths = [[('user', 'eats', 'food'), ('food', 'eaten', 'user')],
                  [('user', 'has', 'habit'), ('habit', 'from', 'user')],
                  [('user', 'eats', 'food'), ('food', 'contains', 'ingredient'), ('ingredient', 'in', 'food'),
                   ('food', 'eaten', 'user')],
                  [('user', 'eats', 'food'), ('food', 'belongs_to', 'category'), ('category', 'contains', 'food'),
                   ('food', 'eaten', 'user')]]

    graph = AddMetaPaths(meta_paths, drop_orig_edge_types=True,
                         drop_unconnected_node_types=True)(graph)
    torch.manual_seed(seed)
    for metapath in graph.metapath_dict.keys():
        data = graph[metapath].edge_index
        num_columns_to_sample = int(data.size(1) * sample_ratio)
        column_indices = torch.randperm(data.shape[1])[:num_columns_to_sample]
        sampled_data = data[:, column_indices]
        graph[metapath].edge_index = sampled_data.clone()

    return graph


def process_features_for_MLP(graph):
    user2food = generate_edge_mapping(graph, ('user', 'eats', 'food'))
    user2habit = generate_edge_mapping(graph, ('user', 'has', 'habit'))
    food2ingredient = generate_edge_mapping(graph, ('food', 'contains', 'ingredient'))
    food2category = generate_edge_mapping(graph, ('food', 'belongs_to', 'category'))

    features = []
    for user_id in range(len(graph['user'].node_id)):
        food_ids = user2food[user_id]
        habit_ids = user2habit[user_id]
        ingredient_ids = [ingredient_id for food_id in food_ids if food_id in food2ingredient
                          for ingredient_id in food2ingredient[food_id]]
        category_ids = [category_id for food_id in food_ids if food_id in food2category
                        for category_id in food2category[food_id]]
        # Initialize food_emb and habit_emb as PyTorch tensors
        food_emb = torch.stack([graph['food'].x[food_id] for food_id in food_ids]) if food_ids else torch.empty(0)
        habit_emb = torch.stack([graph['habit'].x[habit_id] for habit_id in habit_ids]) if habit_ids else torch.empty(0)
        ingredient_emb = torch.stack([graph['ingredient'].x[ingredient_id] for ingredient_id in
                                      ingredient_ids]) if ingredient_ids else torch.empty(0)
        category_emb = torch.stack(
            [graph['category'].x[category_id] for category_id in category_ids]) if category_ids else torch.empty(0)

        # Compute mean embeddings, handling empty cases
        food_emb_mean = torch.mean(food_emb, dim=0) if food_emb.nelement() != 0 else torch.zeros_like(
            graph['user'].x[user_id])
        habit_emb_mean = torch.mean(habit_emb, dim=0) if habit_emb.nelement() != 0 else torch.zeros_like(
            graph['user'].x[user_id])
        ingredient_emb_mean = torch.mean(ingredient_emb, dim=0) if ingredient_emb.nelement() != 0 else torch.zeros_like(
            graph['user'].x[user_id])
        category_emb_mean = torch.mean(category_emb, dim=0) if category_emb.nelement() != 0 else torch.zeros_like(
            graph['user'].x[user_id])

        # Concatenate user embedding with mean food and habit embeddings
        user_feature = torch.cat([food_emb_mean, habit_emb_mean, ingredient_emb_mean, category_emb_mean])
        features.append(user_feature)

    # Stack features outside the loop
    max_length = max(tensor.size(0) for tensor in features)
    # Pad each tensor to the maximum length and collect them in a new list
    padded_features = [F.pad(tensor, (0, max_length - tensor.size(0))) for tensor in features]
    # Stack features outside the loop
    features = torch.stack(padded_features)
    labels = graph['user'].y
    return features, labels


def concat_data_across_years(data_type, file_code, years, year_char):
    """
    This function concat data across years.

    :param data_type: The NHANES data type (demographic, dietary, examination, laboratory, questionnaire)
    :param file_code: the NHANES code of the file.
    :param years: A list of strings identifies which year the data is from. E.g. ['0102', '0304', ..., '1314']
    :param year_char: The char NHANES data used to identify years. This char attaches to the file names as suffix.
        For 01-02 data, the char is B; For 03-04 data, the char is C.
    :return: Returns a pandas dataframe of the concatenated data
    """
    df = pd.DataFrame()
    root_path = '../data/'
    for year in years:
        if year == '1720':
            path = root_path + year + '/' + data_type + '/P_' + file_code + '.XPT'
        else:
            path = root_path + year + '/' + data_type + '/' + file_code + '_' + year_char + '.XPT'

        df_temp = pd.read_sas(path, encoding='ISO-8859-1')
        year_char = chr(ord(year_char) + 1)
        # Record which year the data comes from
        df_temp['years'] = year

        if df.empty:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp])

    # pandas read_sas has an issue to read 0 as 5.397e-79 ...
    df.replace(5.397605346934028e-79, 0, inplace=True)
    return df


def check_coverage(df_main, df_target):
    """
    Most of NHANES data don't have coverage of all respondents. This function tells you what percentage
    of records are covered in the target table of each label group.
    :param df_main: Pandas Dataframe. The main table contains all respondents indexed by SEQN.
    :param df_target: Pandas Dataframe. The target table that only a part of respondents have records.
                    The table is also indexed by SEQN.
    :return: Return a value count pandas dataframe that tells what percentage of records in the main table
            shows up in the target table in each label group.
    """
    df_label = df_main[['label']]
    df_label['exist'] = df_label.index.isin(df_target.index)
    df_exist = df_label.loc[df_label['exist'] == True]

    return round(df_exist['label'].value_counts() / df_main['label'].value_counts(), 4) * 100


def get_food_candidates(data, user_id, top_k=100):
    """
    This function first iterate the food items that a user has eaten. Then for each food item, it finds the similar food
    items if the item that shares the same food category and at least one ingredient. Finally, it calculates the cosine
    similarity between the food item and its similar food items, and return the top k similar food items.

    :param data: The heterogeneous graph data.
    :param user_id: The SEQN of the user.
    :param top_k: The number of food candidates to generate.
    :return: the top k food candidate index and their features.
    """
    # Step 1: Retrieve all the food embeddings that a user has eaten. This is the ground truth.
    user_index = (data['user'].node_id == user_id).nonzero()[0].item()
    food_eaten_index = data[('user', 'eats', 'food')].edge_index[1][
        data[('user', 'eats', 'food')].edge_index[0] == user_index]
    food_eaten = data['food'].x[food_eaten_index]

    # Initialize a priority queue to store the similar food items for each eaten food
    all_similar_pq = []
    for food_index, food_feature in zip(food_eaten_index, food_eaten):
        similar_food_set = set()

        # Find the categories and ingredients of this food item
        categories = data[('food', 'belongs_to', 'category')].edge_index[1][
            data[('food', 'belongs_to', 'category')].edge_index[0] == food_index]
        ingredients = data[('food', 'contains', 'ingredient')].edge_index[1][
            data[('food', 'contains', 'ingredient')].edge_index[0] == food_index]

        # Identify similar food items based on category and ingredient
        for category in categories:
            similar_by_category = data[('food', 'belongs_to', 'category')].edge_index[0][
                data[('food', 'belongs_to', 'category')].edge_index[1] == category]
            for candidate_food in similar_by_category:
                candidate_ingredients = data[('food', 'contains', 'ingredient')].edge_index[1][
                    data[('food', 'contains', 'ingredient')].edge_index[0] == candidate_food]
                if len(set(candidate_ingredients.tolist()).intersection(set(ingredients.tolist()))) > 0:
                    similar_food_set.add(candidate_food.item())

        # Calculate similarities for each similar food
        for similar_food in similar_food_set:
            candidate_feature = data['food'].x[similar_food]
            similarity = cosine_similarity(food_feature.unsqueeze(0), candidate_feature.unsqueeze(0), dim=1)
            heapq.heappush(all_similar_pq, (similarity.item(), similar_food))

    # Remove duplicates. These duplicates could indicate a user had the same food multiple times
    # But we don't consider this scenario for now.
    all_similar_pq = list(set(all_similar_pq))
    # Retrieve the most similar food items and their features, up to 100 if available
    num_items = min(top_k, len(all_similar_pq))
    top_similar_food = [item[1] for item in heapq.nlargest(num_items, all_similar_pq)]
    top_similar_food_features = data['food'].x[top_similar_food]
    top_similar_food_id = data['food'].node_id[top_similar_food]

    return top_similar_food_id, top_similar_food_features


def onehot_encoding(df, categorical_columns):
    """
    :param df: The data frame to be one-hot encoded.
    :param categorical_columns: The columns to be one-hot encoded. The rest of the columns will be left untouched.
    :return: Return a one-hot encoded data frame.
    """
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns),
                              index=df.index)
    # Concatenate the one-hot encoded columns with the original DataFrame
    rest_columns = list(set(df.columns.tolist()) - set(categorical_columns))
    df_encoded = pd.concat([df[rest_columns], encoded_df], axis=1)

    return df_encoded


def check_positive_ratio(graph, edge_list):
    total_match = 0
    for source, target in tqdm(edge_list):
        if graph['user'].y[source] == graph['user'].y[target]:
            total_match += 1
    return round(total_match / len(edge_list), 4)


def generate_edge_mapping(graph, edge):
    mapping = {}
    for src, dst in graph[edge].edge_index.T.tolist():
        if src in mapping:
            mapping[src].append(dst)
        else:
            mapping[src] = [dst]
    return mapping


def generate_neighbors(graph, edge1, edge2, shared_threshold=2):
    user2target = generate_edge_mapping(graph, edge1)
    target2user = generate_edge_mapping(graph, edge2)

    visited_users = set()
    new_edge_list = []
    print("Sampling Neighbors...")
    for user in tqdm(range(len(graph['user'].node_id.tolist()))):
        targets = user2target[user]
        shared_count = defaultdict(int)
        for target in list(set(targets)):
            new_users = target2user[target]
            for new_user in list(set(new_users)):
                if new_user == user or new_user in visited_users:
                    continue
                shared_count[(user, new_user)] += 1

        for user_pair, count in shared_count.items():
            if count >= shared_threshold:
                new_edge_list.append(list(user_pair))
        visited_users.add(user)

    return new_edge_list


def edge_concat(edge_list1, edge_list2):
    # concatenate two edge lists and deduplicate.
    edge_source, edge_target = [], []
    visited_pairs = set()
    print("Concatenating edge lists...")
    for src, tgt in tqdm(edge_list1):
        visited_pairs.add((src, tgt))
        edge_source.append(src)
        edge_target.append(tgt)

    for src, tgt in tqdm(edge_list2):
        if (src, tgt) not in visited_pairs:
            edge_source.append(src)
            edge_target.append(tgt)

    source_edge_index_tensor = torch.tensor(np.array(edge_source), dtype=torch.int64)
    target_edge_index_tensor = torch.tensor(np.array(edge_target), dtype=torch.int64)
    edge_index = torch.vstack([source_edge_index_tensor, target_edge_index_tensor])

    return edge_index


def split_edges(graph, edge_index, test_val_size=0.8):
    """
    split them into train/val/test, and remove the edges that connect across the splits
    :param graph:
    :param edge_index:
    :return: Split edge index
    """
    user_labels = graph['user'].y
    train_indices, val_indices, test_indices, train_labels, val_labels, test_labels, _ = \
        set_split(user_labels, balanced_test=False, test_val_size=test_val_size)

    train_src_edges = []
    train_tgt_edges = []
    val_src_edges = []
    val_tgt_edges = []
    test_src_edges = []
    test_tgt_edges = []
    for src, tgt in tqdm(edge_index.T.tolist()):
        if src in train_indices and tgt in train_indices:
            train_src_edges.append(src)
            train_tgt_edges.append(tgt)
        elif src in val_indices and tgt in val_indices:
            val_src_edges.append(src)
            val_tgt_edges.append(tgt)
        elif src in test_indices and tgt in test_indices:
            test_src_edges.append(src)
            test_tgt_edges.append(tgt)
        else:
            continue

    train_src_edges = torch.tensor(np.array(train_src_edges), dtype=torch.int32)
    train_tgt_edges = torch.tensor(np.array(train_tgt_edges), dtype=torch.int32)
    val_src_edges = torch.tensor(np.array(val_src_edges), dtype=torch.int32)
    val_tgt_edges = torch.tensor(np.array(val_tgt_edges), dtype=torch.int32)
    test_src_edges = torch.tensor(np.array(test_src_edges), dtype=torch.int32)
    test_tgt_edges = torch.tensor(np.array(test_tgt_edges), dtype=torch.int32)
    train_edge_index = torch.vstack([train_src_edges, train_tgt_edges])
    val_edge_index = torch.vstack([val_src_edges, val_tgt_edges])
    test_edge_index = torch.vstack([test_src_edges, test_tgt_edges])

    return train_edge_index, val_edge_index, test_edge_index


def extract_subgraph(graph, user_pair):
    subgraph = HeteroData()
    user_indices = user_pair.tolist()

    subgraph['user'].x = graph['user'].x[user_indices]
    subgraph['user'].node_id = graph['user'].node_id[user_indices]
    subgraph.y = 1 if graph['user'].y[user_indices[0]] == graph['user'].y[user_indices[1]] else 0

    new_user_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(user_indices)}
    for relation in [('user', 'eats', 'food'), ('user', 'has', 'habit')]:
        user2target = graph[relation].edge_index

        mask = torch.isin(user2target[0], user_pair)
        filtered_edge_index = user2target[:, mask]

        target_indices = torch.unique(user2target[:, mask][1])
        new_target_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(target_indices.tolist())}

        # Remap target node indices to new indices
        remapped_target_indices = torch.tensor([new_target_indices[idx.item()] for idx in filtered_edge_index[1]])
        remapped_user_indices = torch.tensor([new_user_indices[idx.item()] for idx in filtered_edge_index[0]])

        subgraph[relation].edge_index = torch.stack([remapped_user_indices, remapped_target_indices], dim=0)

        reverse_relation = (relation[2], relation[1] + '_reverse', relation[0])
        subgraph[reverse_relation].edge_index = torch.stack([remapped_target_indices, remapped_user_indices], dim=0)

        subgraph[relation[2]].x = graph[relation[2]].x[target_indices]
        subgraph[relation[2]].node_id = graph[relation[2]].node_id[target_indices]

    return subgraph


def sample_subgraph(graph, edge_index, is_sample=True, sample_size=1000):
    if is_sample:
        indices = torch.randperm(edge_index.size(1))[:sample_size]
        edge_index = edge_index[:, indices]
    subgraph_list = []
    print("Sampling...")
    for user_pair in tqdm(edge_index.t()):
        subgraph_list.append(extract_subgraph(graph, user_pair))

    return subgraph_list


class GCDataset(Dataset):
    def __init__(self, graph_path='../processed_data/heterogeneous_graph_768_no_med_balanced_with_metapath.pt',
                 is_sample=True, sample_size=1000, food_threshold=5, habit_threshold=8):
        super(GCDataset, self).__init__()
        self.is_sample = is_sample
        self.sample_size = sample_size
        self.food_threshold = food_threshold
        self.habit_threshold = habit_threshold
        self.graph = torch.load(graph_path)

        self.data_list = self.process()

    def process(self):
        UFU_edge_list = generate_neighbors(self.graph, ('user', 'eats', 'food'), ('food', 'eaten', 'user'),
                                           self.food_threshold)
        UHU_edge_list = generate_neighbors(self.graph, ('user', 'has', 'habit'), ('habit', 'from', 'user'),
                                           self.habit_threshold)
        edge_index = edge_concat(UFU_edge_list, UHU_edge_list)
        train_edge_index, val_edge_index, test_edge_index = split_edges(self.graph, edge_index)

        train_subgraph_list = sample_subgraph(self.graph, train_edge_index, is_sample=self.is_sample,
                                              sample_size=self.sample_size)
        valid_subgraph_list = sample_subgraph(self.graph, val_edge_index, is_sample=self.is_sample,
                                              sample_size=self.sample_size * 2)
        test_subgraph_list = sample_subgraph(self.graph, test_edge_index, is_sample=self.is_sample,
                                             sample_size=self.sample_size * 2)

        return train_subgraph_list, valid_subgraph_list, test_subgraph_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_labels(graph, edge_index):
    labels = []
    for i in range(edge_index.T.shape[0]):
        labels.append(1 if graph['user'].y[edge_index.t()[i, 0]] == graph['user'].y[edge_index.t()[i, 1]] else 0)
    return torch.tensor(labels)


"""
Functions for data splitting
"""


def create_balanced_set(indices, labels):
    """
    :param indices: The indices of the set
    :param labels: The labels of the set
    :return: The funtion takes in an imbalanced set, keep all the positive labels, and sample equal amount of negative
            samples, and returns a balanced set of indices and labels
    """
    # Identify positive and negative indices in the test set
    positive_indices = indices[labels == 1]
    negative_indices = indices[labels == 0]
    # Determine the size of the positive class in the test set
    positive_class_size = len(positive_indices)
    # Randomly sample an equal number of negative samples for the test set
    balanced_negative_indices = np.random.choice(negative_indices, size=positive_class_size, replace=False)
    # Combine to form the balanced test set
    balanced_indices = np.concatenate([balanced_negative_indices, positive_indices])
    # Create the corresponding balanced test labels
    balanced_labels = np.concatenate([np.zeros(positive_class_size), np.ones(positive_class_size)])

    return balanced_indices, balanced_labels


def get_weights(train_labels):
    """
    :param train_labels: The labels of the training set.
    :return: This function will calculate the weight of each class in the training set, and return a tensor of weights.
            This is used for re-weighting the loss function to deal with imbalanced data.
    """
    num_positive = train_labels.sum().item()
    num_negative = len(train_labels) - num_positive

    weight_for_0 = 1. / num_negative
    weight_for_1 = 1. / num_positive
    weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)

    return weights


def set_split(labels, balanced_test=True, test_val_size=0.8, test_val_split=0.5):
    # Generate indices for all the user nodes
    all_indices = torch.arange(len(labels))
    train_indices, test_indices, train_labels, test_labels = train_test_split(
        all_indices.numpy(), labels.numpy(), test_size=test_val_size, random_state=42, stratify=labels.numpy())

    val_indices, test_indices, val_labels, test_labels = train_test_split(
        test_indices, test_labels, test_size=test_val_split, random_state=42, stratify=test_labels)

    weights = get_weights(train_labels)
    if balanced_test:
        val_indices, val_labels = create_balanced_set(val_indices, val_labels)
        test_indices, test_labels = create_balanced_set(test_indices, test_labels)

    train_labels = torch.LongTensor(train_labels)
    val_labels = torch.LongTensor(val_labels)
    test_labels = torch.LongTensor(test_labels)

    return train_indices, val_indices, test_indices, train_labels, val_labels, test_labels, weights


"""
Data Loading Functions
"""


def load_data_for_prompt_RGCN(graph):
    features = torch.cat([graph['user'].prompt_embedding, graph['food'].prompt_embedding,
                          graph['ingredient'].prompt_embedding, graph['category'].prompt_embedding,
                          graph['habit'].prompt_embedding], dim=0)
    pca = PCA(n_components=512)
    features = pca.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    node_feature_dims = features.shape[1]

    offset_dict = {
        'user': 0,
        'food': graph['user'].x.shape[0],
        'ingredient': graph['user'].x.shape[0] + graph['food'].x.shape[0],
        'category': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0],
        'habit': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0] + \
                 graph['category'].x.shape[0]
    }

    user_food = get_offset_edges(graph[('user', 'eats', 'food')].edge_index, offset_dict, 'user', 'food')
    food_ingredient = get_offset_edges(graph[('food', 'contains', 'ingredient')].edge_index, offset_dict, 'food',
                                       'ingredient')
    food_category = get_offset_edges(graph[('food', 'belongs_to', 'category')].edge_index, offset_dict, 'food',
                                     'category')
    user_habit = get_offset_edges(graph[('user', 'has', 'habit')].edge_index, offset_dict, 'user', 'habit')

    # Concatenate edge indices and types
    edge_index_all = torch.cat([
        user_food,
        food_ingredient,
        food_category,
        user_habit
    ], dim=1)

    # Create edge types
    # Assuming 0 for 'user, ate, food', 1 for 'food, contains, ingredient', and 2 for 'food, belongs_to, category'
    edge_type_all = torch.cat([
        torch.zeros(graph[('user', 'eats', 'food')].edge_index.shape[1], dtype=torch.long),
        torch.ones(graph[('food', 'contains', 'ingredient')].edge_index.shape[1], dtype=torch.long),
        2 * torch.ones(graph[('food', 'belongs_to', 'category')].edge_index.shape[1], dtype=torch.long),
        3 * torch.ones(graph[('user', 'has', 'habit')].edge_index.shape[1], dtype=torch.long)
    ])

    # Number of relations (edge types)
    num_relations = edge_type_all.max().item() + 1
    user_labels = graph['user'].y

    return node_feature_dims, features, edge_index_all, edge_type_all, num_relations, user_labels


def load_data_for_RGCN(graph):
    node_feature_dims = {
        'user': graph['user'].x.shape[1],
        'food': graph['food'].x.shape[1],
        'ingredient': graph['ingredient'].x.shape[1],
        'category': graph['category'].x.shape[1],
        'habit': graph['habit'].x.shape[1]
    }
    feature_dict = {
        'user': graph['user'].x,
        'food': graph['food'].x,
        'ingredient': graph['ingredient'].x,
        'category': graph['category'].x,
        'habit': graph['habit'].x
    }

    offset_dict = {
        'user': 0,
        'food': graph['user'].x.shape[0],
        'ingredient': graph['user'].x.shape[0] + graph['food'].x.shape[0],
        'category': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0],
        'habit': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0] + \
                 graph['category'].x.shape[0]
    }

    user_food = get_offset_edges(graph[('user', 'eats', 'food')].edge_index, offset_dict, 'user', 'food')
    food_ingredient = get_offset_edges(graph[('food', 'contains', 'ingredient')].edge_index, offset_dict, 'food',
                                       'ingredient')
    food_category = get_offset_edges(graph[('food', 'belongs_to', 'category')].edge_index, offset_dict, 'food',
                                     'category')
    user_habit = get_offset_edges(graph[('user', 'has', 'habit')].edge_index, offset_dict, 'user', 'habit')

    # Concatenate edge indices and types
    edge_index_all = torch.cat([
        user_food,
        food_ingredient,
        food_category,
        user_habit
    ], dim=1)

    # Create edge types
    # Assuming 0 for 'user, ate, food', 1 for 'food, contains, ingredient', and 2 for 'food, belongs_to, category'
    edge_type_all = torch.cat([
        torch.zeros(graph[('user', 'eats', 'food')].edge_index.shape[1], dtype=torch.long),
        torch.ones(graph[('food', 'contains', 'ingredient')].edge_index.shape[1], dtype=torch.long),
        2 * torch.ones(graph[('food', 'belongs_to', 'category')].edge_index.shape[1], dtype=torch.long),
        3 * torch.ones(graph[('user', 'has', 'habit')].edge_index.shape[1], dtype=torch.long)
    ])

    # Number of relations (edge types)
    num_relations = edge_type_all.max().item() + 1
    user_labels = graph['user'].y

    return node_feature_dims, feature_dict, edge_index_all, edge_type_all, num_relations, user_labels


def get_offset_edges(edge_index, offset_dict, left_name, right_name):
    src_nodes, dst_nodes = edge_index
    src_nodes_offset = src_nodes + offset_dict[left_name]
    dst_nodes_offset = dst_nodes + offset_dict[right_name]
    return torch.stack([src_nodes_offset, dst_nodes_offset])


def load_data_for_HAN(graph):
    """
    TODO: This is broken. Need to fix.
    """
    node_feature_dims = {
        'user': graph['user'].x.shape[1],
        'food': graph['food'].x.shape[1],
        'ingredient': graph['ingredient'].x.shape[1],
        'category': graph['category'].x.shape[1],
        'habit': graph['habit'].x.shape[1]
    }
    feature_dict = {
        'user': graph['user'].x,
        'food': graph['food'].x,
        'ingredient': graph['ingredient'].x,
        'category': graph['category'].x,
        'habit': graph['habit'].x
    }

    offset_dict = {
        'user': 0,
        'food': graph['user'].x.shape[0],
        'ingredient': graph['user'].x.shape[0] + graph['food'].x.shape[0],
        'category': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0],
        'habit': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0] + \
                 graph['category'].x.shape[0]
    }
    user_habit_adj = edge_list_to_adj_matrix(graph[('user', 'has', 'habit')], graph['user'].x.shape[0],
                                             graph['habit'].x.shape[0], 'user', 'habit', offset_dict)
    food_category_adj = edge_list_to_adj_matrix(graph[('food', 'belongs_to', 'category')], graph['food'].x.shape[0],
                                                graph['category'].x.shape[0], 'food', 'category', offset_dict)
    food_ingredient_adj = edge_list_to_adj_matrix(graph[('food', 'contains', 'ingredient')], graph['food'].x.shape[0],
                                                  graph['ingredient'].x.shape[0], 'food', 'ingredient', offset_dict)
    user_food_adj = edge_list_to_adj_matrix(graph[('user', 'eats', 'food')], graph['user'].x.shape[0],
                                            graph['food'].x.shape[0], 'user', 'food', offset_dict)

    # Meta-path: User - Habit - User
    UHU = torch.sparse.mm(user_habit_adj, user_habit_adj.t())
    # Meta-path: User - Food - User
    UFU = torch.mm(user_food_adj, user_food_adj.t())
    # Meta-path: User - Food - Ingredient - Food - User
    food_ingredient_food_adj = torch.mm(food_ingredient_adj, food_ingredient_adj.t())
    UFIFU = torch.mm(torch.mm(user_food_adj, food_ingredient_food_adj),
                     user_food_adj.t())
    # Meta-path: User - Food - Category - Food - User
    food_category_food = torch.mm(food_category_adj, food_category_adj.t())
    UFCFU = torch.mm(torch.mm(user_food_adj, food_category_food),
                     user_food_adj.t())
    meta_path_list = [UHU, UFU, UFIFU, UFCFU]

    user_labels = graph['user'].y

    return node_feature_dims, feature_dict, meta_path_list, user_labels


def edge_list_to_adj_matrix(edge_list, left_num, right_num, left_name, right_name, offset_dict):
    adj_matrix = torch.zeros(left_num, right_num, dtype=torch.long)
    src_nodes, dst_nodes = edge_list

    # Assuming offset is subtracting the offset value
    src_nodes_offset = src_nodes - offset_dict[left_name]
    dst_nodes_offset = dst_nodes - offset_dict[right_name]

    adj_matrix[src_nodes_offset, dst_nodes_offset] += 1
    return adj_matrix


def load_data_for_homo_refined(graph):
    node_features = graph['user'].x
    edge_index = torch.cat((graph[('user', 'UFU', 'user')].edge_index, graph[('user', 'UHU', 'user')].edge_index), dim=1)
    user_labels = graph['user'].y

    return node_features, edge_index, user_labels



def load_data_for_GCN(graph, data_type='feature'):
    if data_type == 'feature':
        D = graph['food'].x.shape[1]  # We will pad the features to the max feature dimension
        for node_type in ['user', 'food', 'ingredient', 'category', 'habit']:
            pad_length = D - graph[node_type]['x'].shape[1]
            graph[node_type]['x'] = torch.tensor(np.pad(graph[node_type]['x'], ((0, 0), (0, pad_length)), 'constant'))
        # Again the sequence is very important.
        node_features = torch.cat(
            [graph[node_type]['x'] for node_type in ['user', 'food', 'ingredient', 'category', 'habit']])
    elif data_type == 'prompt':
        node_features = torch.cat(
            [graph[node_type]['prompt_embedding'] for node_type in ['user', 'food', 'ingredient', 'category', 'habit']])
        pca = PCA(n_components=512)
        node_features = pca.fit_transform(node_features)
        node_features = torch.tensor(node_features, dtype=torch.float32)
    else:
        raise NotImplementedError('data_type must be either feature or prompt')

    offset_dict = {
        'user': 0,
        'food': graph['user'].x.shape[0],
        'ingredient': graph['user'].x.shape[0] + graph['food'].x.shape[0],
        'category': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0],
        'habit': graph['user'].x.shape[0] + graph['food'].x.shape[0] + graph['ingredient'].x.shape[0] + \
                 graph['category'].x.shape[0]
    }

    user_food_edges = get_offset_edges(graph[('user', 'eats', 'food')].edge_index, offset_dict, 'user', 'food')
    food_ingredient_edges = get_offset_edges(graph[('food', 'contains', 'ingredient')].edge_index, offset_dict, 'food',
                                             'ingredient')
    food_category_edges = get_offset_edges(graph[('food', 'belongs_to', 'category')].edge_index, offset_dict, 'food',
                                           'category')
    user_habit_edges = get_offset_edges(graph[('user', 'has', 'habit')].edge_index, offset_dict, 'user', 'habit')

    # Concatenate all edges
    edge_index = torch.cat((user_food_edges, food_ingredient_edges, food_category_edges, user_habit_edges), dim=1)
    user_labels = graph['user'].y

    return node_features, edge_index, user_labels


"""
User Health Status Tagging
"""


def tag_BMI_waist_circumference(row):
    underweight_BMI, overweight_BMI = 18.5, 25

    waist_threshold_male, waist_threshold_female = 102, 88
    waist_threshold = waist_threshold_male if row['gender'] == '1' else waist_threshold_female

    high_calories, low_calories = 0, 0
    if row['BMXBMI'] < underweight_BMI:
        high_calories = 1
    if row['BMXBMI'] >= overweight_BMI or row['BMXWAIST'] >= waist_threshold:
        low_calories = 1

    # this rarely happens, but it means the visceral fat is high, so the user still need low calories food.
    if high_calories == 1 and low_calories == 1:
        high_calories = 0

    return high_calories, low_calories


def tag_blood_pressure(row):
    high_systolic_threshold, high_diastolic_threshold = 120, 80
    low_sodium, high_potassium = 0, 0

    # Check if the blood pressure is above the thresholds
    if row['Average_Systolic'] >= high_systolic_threshold or row['Average_Diastolic'] >= high_diastolic_threshold:
        low_sodium = 1
        high_potassium = 1

    return low_sodium, high_potassium


"""
Prompt Engineering
"""


def user_prompt_adding(row):
    gender = 'male' if row['gender'] == '1' else 'female'
    race_dict = {
        '0': 'Missing',
        '1': 'Mexican American',
        '2': 'Other Hispanic',
        '3': 'White',
        '4': 'Black',
        '5': 'Other race'
    }
    education_dict = {
        '0': 'Missing',
        '1': 'Less than 9th grade',
        '2': '9-11th grade',
        '3': 'GED or equivalent',
        '4': 'Some college or AA degree',
        '5': 'College graduate or above',
        '7': 'Refused',
        '9': "Don't know"
    }

    return f"User Node: {gender}, age {row['age']}, {race_dict[row['race']]}, " \
           f"household income level {row['household_income']}, education status: {education_dict[row['education']]}"


def food_prompt_adding(row, nutrition_columns):
    return f"Food Node: The food description is: {row['food_desc']}, the nutrition component vector is [{row[nutrition_columns].values.tolist()}]."


def ingredient_prompt_adding(row):
    return f"Ingredient Node: The ingredient description is: {row['ingredient_desc']}."


def category_prompt_adding(row):
    return f"Food Category Node: The food category description is: {row['WWEIA_desc']}."


def habit_prompt_adding(row):
    return f"Habit Node: The habit description is: {row['habit_desc']}."


def sp_mat_to_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


def inner_product(x1, x2):
    return torch.sum(torch.mul(x1, x2), dim=-1)


def compute_bpr_loss(x1, x2):
    #return -torch.sum(torch.log((x1.view(-1) - x2.view(-1)).sigmoid() + 1e-8))
    return -torch.sum(F.logsigmoid(x1-x2))

def compute_infoNCE_loss(x1, x2, temp):
    return torch.logsumexp((x2 - x1[:, None]) / temp, dim=1)


def compute_reg_loss(w1, w2, w3):
    return 0.5 * torch.sum(torch.pow(w1, 2) + torch.pow(w2, 2) + torch.pow(w3, 2))