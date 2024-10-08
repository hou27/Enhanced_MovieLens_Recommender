import torch
from torch_geometric.nn import GATConv
from torch.nn import functional as F
from torch.nn import Parameter

from torch_geometric.nn.inits import glorot

class GraphRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GraphRecsysModel, self).__init__()

        self._init(**kwargs)

        self.reset_parameters()

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self, pos_edge_index, neg_edge_index):# pos_neg_pair_t):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(pos_edge_index)
        neg_pred = self.predict(neg_edge_index)
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()

        return cf_loss
    
    def update_graph_input(self, dataset):
        raise NotImplementedError

    def predict(self, unids, inids):
        raise NotImplementedError

    def eval(self, metapath_idx=None):
        super(GraphRecsysModel, self).eval()
        if self.__class__.__name__ not in ['KGATRecsysModel', 'KGCNRecsysModel']:
            if self.__class__.__name__[:3] == 'PEA':
                with torch.no_grad():
                    self.cached_repr = self.forward(metapath_idx)
            else:
                with torch.no_grad():
                    self.cached_repr = self.forward()


class PEABaseChannel(torch.nn.Module):
    def reset_parameters(self):
        for module in self.gnn_layers:
            module.reset_parameters()

    def forward(self, x, edge_index_list):
        assert len(edge_index_list) == self.num_steps

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gnn_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gnn_layers[-1](x, edge_index_list[-1])
        return x
    
class PEABaseRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(PEABaseRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        # self.entity_aware = kwargs['entity_aware']
        # self.entity_aware_coff = kwargs['entity_aware_coff']
        self.meta_path_steps = kwargs['meta_path_steps'].split(",")
        self.if_use_features = False
        self.channel_aggr = kwargs['channel_aggr']

        # Create node embedding
        if not self.if_use_features:
            self.x = Parameter(torch.Tensor(kwargs['dataset'].data['num_nodes'], kwargs['hidden_dim']))
        else:
            raise NotImplementedError('Feature not implemented!')

        # Create graphs
        meta_path_edge_index_list = self.update_graph_input(kwargs['dataset'])
        assert len(meta_path_edge_index_list) == len(self.meta_path_steps)
        self.meta_path_edge_index_list = meta_path_edge_index_list

        # Create channels
        self.pea_channels = torch.nn.ModuleList()
        for num_steps in self.meta_path_steps:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = int(num_steps)
            self.pea_channels.append(kwargs_cpy['channel_class'](**kwargs_cpy))

        if self.channel_aggr == 'att':
            self.att = Parameter(torch.Tensor(1, len(self.meta_path_steps), kwargs['repr_dim']))

        if self.channel_aggr == 'cat':
            self.fc1 = torch.nn.Linear(2 * len(self.meta_path_steps) * kwargs['repr_dim'], kwargs['repr_dim'])
        else:
            self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        for module in self.pea_channels:
            module.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        if self.channel_aggr == 'att':
            glorot(self.att)

    def forward(self, metapath_idx=None):
        x = self.x
        x = [module(x, self.meta_path_edge_index_list[idx]).unsqueeze(1) for idx, module in enumerate(self.pea_channels)]
        if metapath_idx is not None:
            x[metapath_idx] = torch.zeros_like(x[metapath_idx])
        x = torch.cat(x, dim=1)
        if self.channel_aggr == 'concat':
            x = x.view(x.shape[0], -1)
        elif self.channel_aggr == 'mean':
            x = x.mean(dim=1)
        elif self.channel_aggr == 'att':
            atts = F.softmax(torch.sum(x * self.att, dim=-1), dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        return x

    def predict(self, edge_index):# unids, inids):
        # u_repr = self.cached_repr[unids]
        # i_repr = self.cached_repr[inids]

        u_repr = self.cached_repr[edge_index[0]]
        i_repr = self.cached_repr[edge_index[1]]
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1)

class PEAGATChannel(PEABaseChannel):
    def __init__(self, **kwargs):
        super(PEAGATChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        self.gnn_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] == 1:
            self.gnn_layers.append(GATConv(kwargs['hidden_dim'], kwargs['repr_dim'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
        else:
            self.gnn_layers.append(GATConv(kwargs['hidden_dim'], kwargs['hidden_dim'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            for i in range(kwargs['num_steps'] - 2):
                self.gnn_layers.append(GATConv(kwargs['hidden_dim'] * kwargs['num_heads'], kwargs['hidden_dim'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            self.gnn_layers.append(GATConv(kwargs['hidden_dim'] * kwargs['num_heads'], kwargs['repr_dim'], heads=1, dropout=kwargs['dropout']))

        self.reset_parameters()


class PEAGATRecsysModel(PEABaseRecsysModel):
    def __init__(self, **kwargs):
        kwargs['channel_class'] = PEAGATChannel
        super(PEAGATRecsysModel, self).__init__(**kwargs)