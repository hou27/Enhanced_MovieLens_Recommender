import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch import Tensor
from torch_geometric.nn.inits import glorot

def update_pea_graph_input(dataset, train_args):
    if dataset.name == "Movielens":
        data = dataset.data
        device = train_args['device']
        
        user2item_edge_index = data['user', 'rates', 'movie']['edge_index'].to(device)
        item2user_edge_index = user2item_edge_index.flip([0])
        item2year_edge_index = data['movie', 'in', 'year']['edge_index'].to(device)
        year2item_edge_index = item2year_edge_index.flip([0])
        item2genre_edge_index = data['movie', 'of', 'genre']['edge_index'].to(device)
        genre2item_edge_index = item2genre_edge_index.flip([0])
        tag2item_edge_index = data['tag', 'describes', 'movie']['edge_index'].to(device)
        item2tag_edge_index = tag2item_edge_index.flip([0])
        tag2user_edge_index = data['tag', 'used_by', 'user']['edge_index'].to(device)
        user2tag_edge_index = tag2user_edge_index.flip([0])

        # 메타 패스 정의
        meta_path_edge_indices = [
            # 사용자 중심 메타 패스
            [user2item_edge_index, item2user_edge_index],  # U-I-U
            [year2item_edge_index, item2user_edge_index], # Y-I-U
            [genre2item_edge_index, item2user_edge_index],  # G-I-U
            [user2tag_edge_index, tag2user_edge_index],  # U-T-U
            [item2tag_edge_index, tag2user_edge_index],  # I-T-U
            
            # 아이템(영화) 중심 메타 패스
            [item2user_edge_index, user2item_edge_index],  # I-U-I
            # [item2year_edge_index, year2item_edge_index],  # I-Y-I
            # [item2genre_edge_index, genre2item_edge_index],  # I-G-I
            [item2tag_edge_index, tag2item_edge_index],  # I-T-I
            [user2tag_edge_index, tag2item_edge_index],  # U-T-I
            [tag2user_edge_index, user2item_edge_index],  # T-U-I
        ]

        # print("Print sample meta path edge indices:")
        # for idx, edge_indices in enumerate(meta_path_edge_indices):
        #     print(f"Meta path {idx}: {edge_indices[0].shape} - {edge_indices[1].shape}")
        #     print(f"Meta path {idx}: {edge_indices}")

        if dataset.type == "25m":
            genome_tag2item_edge_index = data['genome_tag', 'describes', 'movie']['edge_index'].to(device)
            item2genome_tag_edge_index = genome_tag2item_edge_index.flip([0])
            meta_path_edge_indices.extend([
                [genome_tag2item_edge_index, item2user_edge_index],  # GT-I-U
                # [item2genome_tag_edge_index, genome_tag2item_edge_index],  # I-GT-I
            ])

        return meta_path_edge_indices
    else:
        raise NotImplementedError("Only Movielens dataset is implemented.")

class PEAGATChannel(nn.Module):
    def __init__(self, in_dim, out_dim, repr_dim, num_steps, num_heads, dropout):
        super().__init__()
        self.num_steps = num_steps
        self.gnn_layers = nn.ModuleList()
        if num_steps > 1:
            self.gnn_layers.append(GATConv(in_dim, out_dim, heads=num_heads, dropout=dropout))
            for _ in range(num_steps - 2):
                self.gnn_layers.append(GATConv(out_dim * num_heads, out_dim, heads=num_heads, dropout=dropout))
            self.gnn_layers.append(GATConv(out_dim, repr_dim, heads=num_heads, dropout=dropout))
        else:
            self.gnn_layers.append(GATConv(in_dim, repr_dim, heads=num_heads, dropout=dropout))

        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.gnn_layers:
            layer.reset_parameters()
    
    def forward(self, x, edge_index_list):
        assert len(edge_index_list) == self.num_steps

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gnn_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gnn_layers[-1](x, edge_index_list[-1])
        return x

class PEAGATRecsysModel(nn.Module):
    def __init__(self, dataset, num_nodes, hidden_dim, repr_dim, num_layers, num_heads, dropout, channel_aggr, meta_path_steps, if_use_features=False, train_args=None):
        super().__init__()
        self.data = dataset.data
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.repr_dim = repr_dim
        self.channel_aggr = channel_aggr
        self.meta_path_steps = meta_path_steps.split(",")
        self.if_use_features = if_use_features
        self.num_layers = num_layers

        if not self.if_use_features:
            self.embeddings = Parameter(torch.Tensor(self.num_nodes, hidden_dim))

        self.meta_path_edge_index_list = update_pea_graph_input(dataset, train_args)
        for i, edge_indices in enumerate(self.meta_path_edge_index_list):
            # print(f"Meta-path {i}:")
            for j, edge_index in enumerate(edge_indices):
                # print(f"  Step {j}: max index = {edge_index.max().item()}, shape = {edge_index.shape}")
                if edge_index.max() >= self.num_nodes:
                    print(f"Warning: Meta-path {i}, Step {j} contains index larger than number of nodes")

        self.pea_channels = nn.ModuleList()
        for num_steps in self.meta_path_steps:
            self.pea_channels.append(PEAGATChannel(hidden_dim, hidden_dim, repr_dim, int(num_steps), num_heads, dropout))

        if channel_aggr == 'att':
            self.att = Parameter(Tensor(1, len(self.meta_path_steps), self.repr_dim))

        if channel_aggr == 'cat':
            self.fc1 = nn.Linear(2 * len(self.meta_path_steps) * self.repr_dim, self.repr_dim)
        else:
            self.fc1 = nn.Linear(2 * self.repr_dim, self.repr_dim)
        self.fc2 = nn.Linear(self.repr_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.embeddings)
        for module in self.pea_channels:
            module.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        if self.channel_aggr == 'att':
            glorot(self.att)

    def forward(self, metapath_idx=None):
        if not self.if_use_features:
            x = self.embeddings
        else:
            raise NotImplementedError("Feature-based model not implemented")

        channel_outputs = [channel(x, path_edges).unsqueeze(1) for channel, path_edges in zip(self.pea_channels, self.meta_path_edge_index_list)]
        
        if metapath_idx is not None:
            channel_outputs[metapath_idx] = torch.zeros_like(channel_outputs[metapath_idx])
        
        x = torch.cat(channel_outputs, dim=1)
        
        if self.channel_aggr == 'concat':
            x = x.view(x.shape[0], -1)
        elif self.channel_aggr == 'mean':
            x = x.mean(dim=1)
        elif self.channel_aggr == 'att':
            atts = F.softmax(torch.sum(x * self.att, dim=-1), dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=1)
        else:
            raise NotImplementedError('Other aggr methods not implemented!')
        
        return x

    def predict(self, edge_index):
        u_repr = self.cache_repr[edge_index[0]]
        i_repr = self.cache_repr[edge_index[1]]
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1)
    
    @staticmethod
    def bpr_loss(pos_preds, neg_preds):
        """
        Bayesian Personalized Ranking loss
        """
        return -torch.log(torch.sigmoid(pos_preds - neg_preds) + 1e-10).sum() # mean()
    
    def loss(self, pos_edge_index, neg_edge_index):
        if self.training:
            self.cache_repr = self.forward()
        pos_pred = self.predict(pos_edge_index)
        neg_pred = self.predict(neg_edge_index)
        
        loss = self.bpr_loss(pos_pred, neg_pred)

        return loss