from RCSYS_utils import *
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SignedConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import Tensor
import torch.optim as optim
from torch_geometric.nn import Linear


class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, layers=3, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.layers = embedding_dim, layers
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)  # e_u^0
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.
        Args: edge_index (SparseTensor): adjacency matrix
        Returns: tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])  # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])  # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)


class GCNModel(nn.Module):
    """GCN Model for graph-based recommendation system."""

    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=2, add_self_loops=False, model_type='gcn'):
        """Initializes GCN Model.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 64.
            num_layers (int, optional): Number of GCN layers. Defaults to 20.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.num_layers = embedding_dim, num_layers
        self.add_self_loops = add_self_loops
        self.model_type = model_type

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in ['user', 'food']:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)

        if model_type == 'GCN':
            self.model_layers = nn.ModuleList([GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)])
        elif model_type == 'GAT':
            self.model_layers = nn.ModuleList([GATConv(embedding_dim, embedding_dim) for _ in range(num_layers)])
        elif model_type == 'SAGE':
            self.model_layers = nn.ModuleList([SAGEConv(embedding_dim, embedding_dim) for _ in range(num_layers)])
        elif model_type == 'MLP':
            self.model_layers = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)])
        else:
            raise ValueError('Unknown model type')

    def forward(self, feature_dict, edge_index: SparseTensor):
        edge_index_norm = gcn_norm(edge_index, add_self_loops=self.add_self_loops)

        # Initial embeddings
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        emb_0 = torch.cat([feature_dict['user'], feature_dict['food']])
        emb_k = emb_0

        # GCN layers
        for model_layer in self.model_layers:
            if self.model_type == 'MLP':
                emb_k = F.relu(model_layer(emb_k))
            else:
                emb_k = F.relu(model_layer(emb_k, edge_index_norm))

        # Split embeddings into users and items
        users_emb_final, items_emb_final = torch.split(emb_k,
                                                       [self.num_users, self.num_items])  # Splits into e_u^K and e_i^K

        # Return e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, users_emb_final, items_emb_final, items_emb_final


class SignedGCN(torch.nn.Module):
    r"""The signed graph convolutional network model from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.
    Internally, this module uses the
    :class:`torch_geometric.nn.conv.SignedConv` operator.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of layers.
        lamb (float, optional): Balances the contributions of the overall
            objective. (default: :obj:`5`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """
    def __init__(self, num_users, num_foods, hidden_channels, num_layers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_foods

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hidden_channels)  # e_u^0
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hidden_channels)  # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        self.conv1 = SignedConv(hidden_channels, hidden_channels // 2,
                                first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SignedConv(hidden_channels // 2, hidden_channels // 2,
                           first_aggr=False))

        self.lin = torch.nn.Linear(2 * hidden_channels, 3)
        self.reset_parameters()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tensor:
        """Computes node embeddings :obj:`z` based on positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`.

        Args:
            x (torch.Tensor): The input node features.
            pos_edge_index (torch.Tensor): The positive edge indices.
            neg_edge_index (torch.Tensor): The negative edge indices.
        """
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            z (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)

        log_softmax_output = torch.log_softmax(value, dim=1)
        class_indices = torch.argmax(log_softmax_output, dim=1)

        # Map class indices to desired values: 0 -> -1, 1 -> 0, 2 -> 1
        mapping = torch.tensor([-1, 0, 1]).to(value.device)
        mapped_output = mapping[class_indices]

        return mapped_output

###
# New Development
###

# This module is fully encapsulated and understood
# This defines a layer with a learnable transformation
class MetricCalculator(nn.Module):
    """Applies a learnable transformation to the node features."""
    def __init__(self, feature_dim):
        super(MetricCalculator, self).__init__()
        self.weight = nn.Parameter(torch.empty((1, feature_dim)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_features):
        return node_features * self.weight

# This module is fully encapsulated and understood
class GraphGenerator(nn.Module):
    """
    Builds a graph based on similarity between node features from two different sets.
    """
    def __init__(self, feature_dim, num_heads=2, similarity_threshold=0.1):
        super(GraphGenerator, self).__init__()
        self.similarity_threshold = similarity_threshold
        self.metric_layers = nn.ModuleList([MetricCalculator(feature_dim) for _ in range(num_heads)])
        self.num_heads = num_heads

    def forward(self, left_features, right_features, edge_index):
        """
        Compute a similarity matrix between left and right node features.
        """
        similarity_matrix = torch.zeros(edge_index.size(1)).to(edge_index.device)
        for metric_layer in self.metric_layers:
            weighted_left = metric_layer(left_features[edge_index[0]])  # users
            weighted_right = metric_layer(right_features[edge_index[1]])  # foods
            similarity_matrix += F.cosine_similarity(weighted_left, weighted_right, dim=1)

        similarity_matrix /= self.num_heads
        return torch.where(similarity_matrix < self.similarity_threshold, torch.zeros_like(similarity_matrix), similarity_matrix)


# This so-called attention layer is simply a weighted layer
class GraphChannelAttLayer(nn.Module):
    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel))
        nn.init.constant_(self.weight, 0.1)  # equal weight

    def forward(self, edge_mask_list):
        edge_mask = torch.stack(edge_mask_list, dim=0)
        # Row normalization of all graphs generated
        edge_mask = F.normalize(edge_mask, dim=1, p=1)
        # Hadamard product + summation -> Conv
        # Apply softmax to the weights to ensure they sum to 1 across channels
        softmax_weights = torch.softmax(self.weight, dim=0)
        
        # Compute the weighted sum of edge masks
        weighted_edge_masks = edge_mask * softmax_weights[:, None]
        
        # Sum over the channel dimension to get the fused edge mask of shape (N)
        fused_edge_mask = torch.sum(weighted_edge_masks, dim=0)

        return fused_edge_mask > 0.5


class SGSL(nn.Module):
    def __init__(self, graph, embedding_dim,  feature_threshold=0.3, num_heads=4, num_layer=3):
        super(SGSL, self).__init__()

        self.num_users = graph['user'].num_nodes
        self.num_foods = graph['food'].num_nodes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layer = num_layer
        self.feature_threshold = feature_threshold

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)
        
        # Graph generators for feature and semantic graphs
        self.feature_graph_generator = GraphGenerator(self.embedding_dim, self.num_heads, self.feature_threshold)
        self.signed_layer = SignedGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer)
        self.fusion = GraphChannelAttLayer(3)
        self.lightgcn = LightGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer, False)


    def forward(self, feature_dict, edge_index, pos_edge_index, neg_edge_index):
        # Heterogeneous Feature Mapping.
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        # Generate the feature graph. The result is a adj_matrix with the same shape as adj_ori
        mask_feature = self.feature_graph_generator(feature_dict['user'], feature_dict['food'], edge_index)
        mask_ori = torch.ones_like(mask_feature)

        # Generate the semantic graph. The same. 
        z = self.signed_layer(pos_edge_index, neg_edge_index)
        mask_semantic = self.signed_layer.discriminate(z, edge_index)

        # Fusion with the original adj with attention 
        edge_mask = self.fusion([mask_ori, mask_feature, mask_semantic])

        edge_index_new = edge_index[:, edge_mask]
        sparse_size = self.num_users + self.num_foods
        sparse_edge_index = SparseTensor(row=edge_index_new[0], col=edge_index_new[1], sparse_sizes=(
            sparse_size, sparse_size))
        
        # Use the new adj, convert back to edge_index, and perform a LightGCN 
        return self.lightgcn(sparse_edge_index)


class modified_SGSL(nn.Module):
    def __init__(self, graph, embedding_dim,  feature_threshold=0.3, num_heads=4, num_layer=3):
        super(modified_SGSL, self).__init__()

        self.num_users = graph['user'].num_nodes
        self.num_foods = graph['food'].num_nodes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layer = num_layer
        self.feature_threshold = feature_threshold

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)
        
        # Graph generators for feature and semantic graphs
        self.feature_graph_generator = GraphGenerator(self.embedding_dim, self.num_heads, self.feature_threshold)
        self.signed_layer = SignedGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer)
        self.fusion = GraphChannelAttLayer(2)
        self.lightgcn = LightGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer, False)


    def forward(self, feature_dict, edge_index, pos_edge_index, neg_edge_index):
        # Heterogeneous Feature Mapping.
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        # Generate the feature graph. The result is a adj_matrix with the same shape as adj_ori
        mask_feature = self.feature_graph_generator(feature_dict['user'], feature_dict['food'], edge_index)
        mask_ori = torch.ones_like(mask_feature)

        # Generate the semantic graph. The same. 
        z = self.signed_layer(pos_edge_index, neg_edge_index)
        mask_semantic = self.signed_layer.discriminate(z, edge_index)

        # # Fusion with the original adj with attention 
        # edge_mask = self.fusion([mask_ori, mask_feature, mask_semantic])

        # # Fusion with the original adj with attention: ablation without b 
        # edge_mask = self.fusion([mask_ori, mask_feature])
        # Fusion with the original adj with attention without a 
        edge_mask = self.fusion([mask_ori, mask_semantic])

        edge_index_new = edge_index[:, edge_mask]
        sparse_size = self.num_users + self.num_foods
        sparse_edge_index = SparseTensor(row=edge_index_new[0], col=edge_index_new[1], sparse_sizes=(
            sparse_size, sparse_size))
        
        # Use the new adj, convert back to edge_index, and perform a LightGCN 
        return self.lightgcn(sparse_edge_index)


class ModifiedGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, graph, num_users, num_items, embedding_dim=64, layers=3, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.layers = embedding_dim, layers
        self.add_self_loops = add_self_loops

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)

        # self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)  # e_u^0
        # self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0

        # nn.init.normal_(self.users_emb.weight, std=0.1)
        # nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, feature_dict, edge_index, share_food=None):
        """Forward propagation of LightGCN Model.
        Args: edge_index (SparseTensor): adjacency matrix
        Returns: tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        print(feature_dict['user'].shape, feature_dict['food'].shape)
        print(self.num_users, self.num_items)
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        edge_index_mod = torch.stack([edge_index[0], edge_index[1] + self.num_users], dim=0)
        sparse_size = self.num_users + self.num_items
        edge_index_mod = SparseTensor(row=edge_index_mod[0], col=edge_index_mod[1], sparse_sizes=(
            sparse_size, sparse_size))
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(edge_index_mod, add_self_loops=self.add_self_loops)
        user_emb = feature_dict['user']
        item_emb = feature_dict['food']

        #user_emb = self.propagate(share_food, x=user_emb)

        emb_0 = torch.cat([user_emb, item_emb])  # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])  # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, users_emb_final, items_emb_final, items_emb_final

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)