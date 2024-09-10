import argparse
from src.models import update_pea_graph_input
import torch

from src import MovieLens, PEAGATRecsysModel, Solver

MODEL = 'PEAGAT'

parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')
parser.add_argument('--type', type=str, default='25m', help='')
parser.add_argument('--num_core', type=int, default=10, help='')

# Model params
parser.add_argument('--hidden_dim', type=int, default=64, help='')
parser.add_argument('--repr_dim', type=int, default=16, help='')
parser.add_argument('--num_layers', type=int, default=2, help='')
parser.add_argument('--num_heads', type=int, default=1, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--channel_aggr', type=str, default='att', help='')
parser.add_argument('--meta_path_steps', type=str, default='2,2,2,2,2,2,2,2,2,2', help='')	# 2,2,2,2,2,2,2,2,2,2,2,2,2(for 25m)

# Train params
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=5, help='')
parser.add_argument('--epochs', type=int, default=30, help='')
parser.add_argument('--batch_size', type=int, default=1024, help='')
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--num_neg_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

args = parser.parse_args()

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = f'cuda:{args.gpu_idx}'

# Setup args
dataset_args = {
    'root': './data', 'name': args.dataset,
    'type': args.type,
    'num_core': args.num_core
}

model_args = {
    'hidden_dim': args.hidden_dim,
    'repr_dim': args.repr_dim,
    'num_layers': args.num_layers,
    'num_heads': args.num_heads,
    'dropout': args.dropout,
    'channel_aggr': args.channel_aggr,
    'meta_path_steps': args.meta_path_steps,
}

train_args = {
    'runs': args.runs,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'num_workers': args.num_workers,
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'device': device,
    'num_neg_samples': args.num_neg_samples,
    'num_neg_candidates': args.num_neg_candidates
}

print('Dataset params:', dataset_args)
print('Model params:', model_args)
print('Train params:', train_args)

class PEAGATRecsysModel(PEAGATRecsysModel):
    def update_graph_input(self, dataset):
        return update_pea_graph_input(dataset_args, train_args, dataset.data)

if __name__ == '__main__':
    dataset = MovieLens(**dataset_args)
    
    # model = PEAGATRecsysModel(dataset.data, **model_args, dataset_args=dataset_args, train_args=train_args)
    # model = PEAGATRecsysModel(**model_args)
    # solver = Solver(model, dataset, train_args)
    solver = Solver(PEAGATRecsysModel, dataset, model_args, train_args)
    solver.run()