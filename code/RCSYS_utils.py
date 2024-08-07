import torch
import random
from sklearn.model_selection import train_test_split
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
from min_norm_solvers import MinNormSolver, gradient_normalizers
from sklearn.metrics import jaccard_score
from torch.nn.functional import cosine_similarity
import torch_geometric

from utils import *


def read_graph_data(path='../processed_data/benchmark_for_recsys.pt'):
    graph = torch.load(path)

    edge_index = graph['user', 'eats', 'food'].edge_index
    user_tags = torch.tensor(graph['user'].tag, dtype=torch.int32)
    food_tags = torch.tensor(graph['food'].tag, dtype=torch.int32)

    mask = []
    for edge in tqdm(edge_index.T):
        src, tgt = edge
        common_ones = user_tags[src] & food_tags[tgt]
        equal_tags = common_ones.sum().item()
        if equal_tags > 0:
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)
    edge_index = edge_index.T[mask].T

    num_users = graph['user'].x.shape[0]
    num_foods = graph['food'].x.shape[0]

    return num_users, num_foods, edge_index


def split_data(edge_index, test_size=0.2, val_size=0.25, seed=42):
    # Split 6-2-2
    edges = edge_index.numpy().T
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=seed)
    train_edges, val_edges = train_test_split(train_edges, test_size=val_size, random_state=seed)

    return train_edges, val_edges, test_edges

def split_data_new(edge_index, edge_label_index, test_size=0.2, val_size=0.25, seed=42):
    # Split 6-2-2
    edges = edge_index.numpy().T
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=seed)
    train_edges, val_edges = train_test_split(train_edges, test_size=val_size, random_state=seed)

    train_edge_index = torch.LongTensor(train_edges).T
    val_edge_index = torch.LongTensor(val_edges).T
    test_edge_index = torch.LongTensor(test_edges).T

    def get_pos_neg_edge_indices(edge_label_index, edge_index):
        # Convert edge_label_index to a set of tuples for easy comparison
        edge_label_set = set([tuple(edge_label_index[:, i].tolist()) for i in range(edge_label_index.size(1))])
        # Identify positive edges in edge_index
        pos_edge_index = torch.tensor([edge for edge in edge_index.t().tolist() if tuple(edge) in edge_label_set]).t()
        # Identify negative edges in edge_index
        neg_edge_index = torch.tensor([edge for edge in edge_index.t().tolist() if tuple(edge) not in edge_label_set]).t()
        return pos_edge_index, neg_edge_index

    # Get positive and negative edges for train, valid, and test sets
    pos_train_edge_index, neg_train_edge_index = get_pos_neg_edge_indices(edge_label_index, train_edge_index)
    pos_val_edge_index, neg_val_edge_index = get_pos_neg_edge_indices(edge_label_index, val_edge_index)
    pos_test_edge_index, neg_test_edge_index = get_pos_neg_edge_indices(edge_label_index, test_edge_index)

    return train_edge_index, val_edge_index, test_edge_index, \
            pos_train_edge_index, neg_train_edge_index, pos_val_edge_index, neg_val_edge_index, \
            pos_test_edge_index, neg_test_edge_index



def sample_mini_batch(batch_size, edge_index, seed=42):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    # torch_geometric.seed_everything(seed)
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, _ = batch[0], batch[1], batch[2]
    # Ensure negative item indices are within valid range
    neg_item_indices = torch.randint(0, int(edge_index[1].max()-1), size=(batch_size,), dtype=torch.long)
    return user_indices, pos_item_indices, neg_item_indices


def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
             lambda_val):
    """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

    Args:
        users_emb_final (torch.Tensor): e_u_k
        users_emb_0 (torch.Tensor): e_u_0
        pos_items_emb_final (torch.Tensor): positive e_i_k
        pos_items_emb_0 (torch.Tensor): positive e_i_0
        neg_items_emb_final (torch.Tensor): negative e_i_k
        neg_items_emb_0 (torch.Tensor): negative e_i_0
        lambda_val (float): lambda value for regularization loss term

    Returns:
        torch.Tensor: scalar bpr loss value
    """
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))  # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # predicted scores of negative samples

    # This encourages the model to rank positive items higher than negative ones
    bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    # Total loss is the sum of BPR loss and regularization
    loss = bpr_loss + reg_loss

    return loss


def jaccard_similarity(user_tags, item_tags):
    intersection = torch.sum(torch.min(user_tags, item_tags), dim=1).float()
    union = torch.sum(torch.max(user_tags, item_tags), dim=1).float()
    jaccard = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    return jaccard


def health_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, user_tags_batch, pos_item_tags_batch, neg_item_tags_batch):
    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # predicted scores of negative samples

    # Calculate cosine similarity between user and food tag vectors
    pos_jaccard = jaccard_similarity(user_tags_batch, pos_item_tags_batch)
    neg_jaccard = jaccard_similarity(user_tags_batch, neg_item_tags_batch)
    jaccard = ((pos_jaccard - neg_jaccard) + 1 ) / 2
  
    # Calculate the health loss
    health_loss = -torch.mean(torch.log(torch.mul(jaccard, torch.sigmoid(pos_scores - neg_scores))))

    return health_loss


def diversity_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, user_features_batch, pos_item_features_batch, neg_item_features_batch, k=20):
    def get_top_k_recommendations(user_emb, item_emb, k=10):
        scores = torch.matmul(user_emb, item_emb.T)
        _, top_k_indices = torch.topk(scores, k=k, dim=1)
        return top_k_indices
    
    def get_mean_similarity(user_features_batch, item_features_batch, k):
        # Get the top K item indices for each user
        top_k_indices = get_top_k_recommendations(user_features_batch, item_features_batch, k)
        top_k_item_embs = item_features_batch[top_k_indices]

        # Calculate the cosine similarities for all pairs in the top K items
        similarities = cosine_similarity(
            top_k_item_embs.unsqueeze(2),  # Shape: (num_users, k, 1, embedding_dim)
            top_k_item_embs.unsqueeze(1),  # Shape: (num_users, 1, k, embedding_dim)
            dim=3
        )

        # Select the upper triangular part of the similarity matrix, excluding the diagonal
        upper_triangular_indices = torch.triu_indices(k, k, 1)
        selected_similarities = similarities[:, upper_triangular_indices[0], upper_triangular_indices[1]]

        # Calculate the mean similarity for each user
        return selected_similarities.mean(dim=1)

    pos_similarity = get_mean_similarity(user_features_batch, pos_item_features_batch, k)
    neg_similarity = get_mean_similarity(user_features_batch, neg_item_features_batch, k)

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)  # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)  # predicted scores of negative samples

    # Calculate and return the diversity loss
    loss = -torch.mean(torch.log((torch.sigmoid(torch.mul(pos_similarity - neg_similarity, pos_scores - neg_scores)))))
    return loss


# helper function to get N_u
def get_user_positive_items(edge_index):
    """Generates dictionary of positive items for each user

    Args:
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        dict: dictionary of positive items for each user
    """
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []

        user_pos_items[user].append(item)
    return user_pos_items


# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    """Computers recall @ k and precision @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute precision and recall on

    Returns:
        tuple: recall @ k, precision @ k
    """
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                   for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()


# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    """Computes Normalized Discounted Cumulative Gain (NDCG) @ k

    Args:
        groundTruth (list): list of lists containing highly rated items of each user
        r (list): list of lists indicating whether each top k item recommended to each user
            is a top k ground truth item or not
        k (int): determines the top k items to compute ndcg on

    Returns:
        float: ndcg @ k
    """
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


def calculate_health_score(users, top_K_items, user_tag, food_tag):
    """Calculates the health score based on the presence of at least one common tag between user and food tags."""
    # Ensure tensors are on the CPU
    user_tags = user_tag[users].cpu()  # Shape: (num_users, num_tags)
    recommended_items = top_K_items[users].cpu()  # Shape: (num_users, k)
    
    # Get the food tags for the top K items for all users
    food_tags = food_tag[recommended_items].cpu()  # Shape: (num_users, k, num_tags)

    # Expand dimensions for broadcasting
    user_tags_expanded = user_tags.unsqueeze(1)  # Shape: (num_users, 1, num_tags)
    
    # Calculate if there's at least one common tag (intersection > 0)
    common_tag = torch.logical_and(user_tags_expanded, food_tags).sum(dim=2) > 0  # Shape: (num_users, k)

    # Calculate the proportion of healthy foods for each user
    healthy_foods_ratio = common_tag.float().mean(dim=1)  # Shape: (num_users)
    
    # Calculate the average health score across all users
    health_score = healthy_foods_ratio.mean().item()
    
    return health_score

def calculate_average_health_tags(users, top_K_items, food_tags):
    """Calculates the average number of health tags each user has been recommended."""
    # Ensure tensors are on the CPU
    recommended_items = top_K_items[users].cpu()  # Shape: (num_users, k)
    
    # Get the food tags for the top K items for all users
    food_tags_recommended = food_tags[recommended_items]  # Shape: (num_users, k, num_tags)
    
    # Sum the tags for each food item for each user
    tags_per_food = food_tags_recommended.sum(dim=2)  # Shape: (num_users, k)
    
    # Average the number of tags per food for each user
    avg_tags_per_user = tags_per_food.mean(dim=1)  # Shape: (num_users,)
    
    # Average across all users
    avg_tags_across_users = avg_tags_per_user.mean().item()

    return avg_tags_across_users

def calculate_percentage_recommended_foods(users, top_K_items, num_foods):
    """Calculates the percentage of foods that have been recommended in top k at least once."""
    recommended_items = top_K_items[users].cpu().flatten().unique()  # Get unique recommended items
    percentage_recommended = len(recommended_items) / num_foods
    
    return percentage_recommended


def get_metrics(model, user_tags, food_tags, edge_index, exclude_edge_indices, k, users_emb_final, items_emb_final):
    """Computes the evaluation metrics: recall, precision, and ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on

    Returns:
        tuple: recall @ k, precision @ k, ndcg @ k
    """
    user_embedding = users_emb_final
    item_embedding = items_emb_final

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)
    # Calculate health score
    health_score = calculate_health_score(users, top_K_items, user_tags, food_tags)

    # Calculate average health tags ratio
    avg_health_tags_ratio = calculate_average_health_tags(users, top_K_items, food_tags)
    
    # Calculate percentage of foods recommended
    num_foods = item_embedding.size(0)
    percentage_recommended_foods = calculate_percentage_recommended_foods(users, top_K_items, num_foods)

    return recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods


# wrapper function to evaluate model
def eval(model, feature_dict, user_tags, food_tags, edge_index, pos_edge_index, neg_edge_index, 
               exclude_edge_indices, k, lambda_val):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # forward pass
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
            model.forward(feature_dict, edge_index, pos_edge_index, neg_edge_index)

    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    # Ensure negative item indices are within valid range
    neg_item_indices = torch.randint(0, int(edge_index[1].max()-1), size=(len(neg_item_indices),), dtype=torch.long)

    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    
    # Don't need it really but just for a rough estimate
    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
        get_metrics(model, user_tags, food_tags, edge_index, exclude_edge_indices, k, users_emb_final, items_emb_final)

    return loss, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods


# wrapper function to evaluate model
def evaluation(model, user_tags, food_tags, edge_index, sparse_edge_index, 
               exclude_edge_indices, k, lambda_val, share_food=None, 
               is_share=False, feature_dict=None, is_feature=True):
    """Evaluates model loss and metrics including recall, precision, ndcg @ k

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges for split to evaluate
        sparse_edge_index (sparseTensor): sparse adjacency matrix for split to evaluate
        exclude_edge_indices ([type]): 2 by N list of edges for split to discount from evaluation
        k (int): determines the top k items to compute metrics on
        lambda_val (float): determines lambda for bpr loss

    Returns:
        tuple: bpr loss, recall @ k, precision @ k, ndcg @ k
    """
    # get embeddings
    if is_share:
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
            feature_dict, sparse_edge_index)
    else:
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
            sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    # Ensure negative item indices are within valid range
    neg_item_indices = torch.randint(0, int(edge_index[1].max()-1), size=(len(neg_item_indices),), dtype=torch.long)

    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    
    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods = \
        get_metrics(model, user_tags, food_tags, edge_index, exclude_edge_indices, k, users_emb_final, items_emb_final)

    return loss, recall, precision, ndcg, health_score, avg_health_tags_ratio, percentage_recommended_foods


def get_user_pos_neg_indices(pos_edge_index, neg_edge_index, node_type, batch_size=2048):
    # Initialize lists to store the results
    user_indices = []
    pos_item_indices = []
    neg_item_indices = []
    offset = (node_type == 0).sum().item()
    # Iterate over unique users in the positive edge index
    for user in pos_edge_index[0].unique():
        # Create masks for the current user in positive and negative edge indices
        user_mask_pos = pos_edge_index[0] == user
        user_mask_neg = neg_edge_index[0] == user

        # Get the item indices corresponding to the current user
        pos_items = pos_edge_index[1][user_mask_pos]
        neg_items = neg_edge_index[1][user_mask_neg].unique()

        # Balance the number of negative items to match the number of positive items
        if len(neg_items) < len(pos_items):
            # If there are fewer negative items, sample additional items randomly to match
            additional_neg_items = torch.randint(offset, pos_edge_index.max().item() + 1, (len(pos_items) - len(neg_items),))
            neg_items = torch.cat([neg_items, additional_neg_items.to(neg_items.device)])
        else:
            # If there are more negative items, truncate to match the number of positive items
            neg_items = neg_items[:len(pos_items)]

        # Store the indices
        user_indices.extend([user] * len(pos_items))
        pos_item_indices.extend(pos_items.tolist())
        neg_item_indices.extend(neg_items.tolist())

    # Convert lists to tensors and ensure the final tensors are of shape (1, N)
    user_indices = torch.tensor(user_indices, dtype=torch.long).unsqueeze(0)
    pos_item_indices = torch.tensor(pos_item_indices, dtype=torch.long).unsqueeze(0)
    neg_item_indices = torch.tensor(neg_item_indices, dtype=torch.long).unsqueeze(0)

    pos_item_indices -= offset
    neg_item_indices -= offset

    idx = torch.randint(0, user_indices.size(0), (batch_size,))

    user_indices = user_indices[:, idx]
    pos_item_indices = pos_item_indices[:, idx]
    neg_item_indices = neg_item_indices[:, idx]

    # The final shape for each tensor in the output is (1, N)
    return user_indices, pos_item_indices, neg_item_indices


def pareto_loss(model, users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, 
                user_features_batch, pos_item_features_batch, neg_item_features_batch, 
                user_tags_batch, pos_item_tags_batch, neg_item_tags_batch, LAMBDA):
            
    loss_data = {}
    grads = {}
    tasks = ['bpr', 'sim', 'health']
    for task in tasks:
        if task == 'bpr':
            loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                    pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)
        elif task == 'sim':
            loss = diversity_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final,
                                  user_features_batch, pos_item_features_batch, neg_item_features_batch)
        elif task == 'health':
            loss = health_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final, 
                                         user_tags_batch, pos_item_tags_batch, neg_item_tags_batch)
        else:
            raise ValueError('Unknown task') 
        loss_data[task] = loss
        grads[task] = []
        loss.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                grads[task].append(param.grad.data.detach().cpu())
        model.zero_grad()
    gn = gradient_normalizers(grads, loss_data, 'l2')
    for task in loss_data:
        for gr_i in range(len(grads[task])):
            grads[task][gr_i] = grads[task][gr_i] / gn[task].to(grads[task][gr_i].device)
    sol, _ = MinNormSolver.find_min_norm_element_FW([grads[task] for task in tasks])
    sol = {k:sol[i] for i, k in enumerate(tasks)}

    model.zero_grad()
    loss = 0
    actual_loss = 0

    for i, l in loss_data.items():
        loss += float(sol[i]) * l
        actual_loss += l
    
    return loss, loss_data, actual_loss