import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")

class Solver:
    def __init__(self, model, dataset, train_args):
        self.model = model
        self.dataset = dataset
        self.train_args = train_args
        self.num_layers = getattr(model, 'num_layers', 2)

    def run(self):
        device = self.train_args['device']
        self.model = self.model.to(device)

        for run in range(self.train_args['runs']):
            self.model.reset_parameters()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_args['lr'],
                                         weight_decay=self.train_args['weight_decay'])

            # print("\n\n\n\n\ntill ok (before first epoch)\n\n\n\n\n")
            for epoch in range(1, self.train_args['epochs'] + 1):
                loss = self.train(optimizer)
                hr10 = self.test()
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, HR@10: {hr10:.4f}')

    def train(self, optimizer):
        self.model.train()
        total_loss = 0

        # 시드 고정 코드 추가
        seed = 2019
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.train_args['batch_size'],
            num_workers=self.train_args['num_workers'],
            collate_fn=self.dataset.collate_fn
        )

        # for batch in loader:
        #     batch = {k: {k2: v2.to(self.train_args['device']) for k2, v2 in v.items()} for k, v in batch.items()}
        #     optimizer.zero_grad()

        #     user_movie_edge_index = batch[('user', 'rates', 'movie')]['edge_index']
        #     user_movie_ratings = batch[('user', 'rates', 'movie')]['edge_attr']

        #     pos_edge_index = user_movie_edge_index[:, user_movie_ratings >= 3]
        #     neg_edge_index = user_movie_edge_index[:, user_movie_ratings < 3]
        #     print(f'pos_edge_index.shape: {pos_edge_index.shape}')
        #     print(f'neg_edge_index.shape: {neg_edge_index.shape}')

        #     loss = self.model.loss(pos_edge_index, neg_edge_index)
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += float(loss) * pos_edge_index.size(1)

        # return total_loss / len(self.dataset)
        
        # positive/negative 샘플 생성 시 양성 샘플과 음성 샘플의 비율을 조정하고, BPR 손실을 계산할 때 이 불균형을 고려
        for batch in loader:
            batch = {k: v.to(self.train_args['device']) for k, v in batch.items()}
            optimizer.zero_grad()

            pos_edge_index = batch['pos_edge_index']
            neg_edge_index = batch['neg_edge_index']

            loss = self.model.loss(pos_edge_index, neg_edge_index)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pos_edge_index.size(1)

        return total_loss / len(self.dataset)

    @torch.no_grad()
    def test(self):
        self.model.eval()

        loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.train_args['batch_size'],
            num_workers=self.train_args['num_workers'],
            collate_fn=self.dataset.collate_fn
        )

        hits = []
        for batch in loader:
            batch = {k: v.to(self.train_args['device']) for k, v in batch.items()}
            
            pos_edge_index = batch['pos_edge_index']
            
            users = pos_edge_index[0].unique()
            for user in users:
                user_pos_movies = pos_edge_index[1][pos_edge_index[0] == user]

                all_movies = torch.arange(self.dataset.data['movie']['num_nodes']).to(self.train_args['device'])
                user_tensor = user.repeat(all_movies.size(0))
                pred = self.model.predict(torch.stack([user_tensor, all_movies]))

                _, indices = torch.sort(pred, descending=True)
                recommended = indices[:10]

                hit = any(item in user_pos_movies for item in recommended)
                hits.append(hit)

        return sum(hits) / len(hits)  # This is HR@10
