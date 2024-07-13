import os
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import psutil
import tqdm

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")

class Solver:
    def __init__(self, model, dataset, train_args):
        self.model = model
        self.dataset = dataset
        self.train_args = train_args
        self.num_layers = getattr(model, 'num_layers', 2)
        self.log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'training_log.jsonl')
        self.best_hr10 = 0

    def run(self):
        device = self.train_args['device']
        self.model = self.model.to(device)

        for run in range(self.train_args['runs']):
            self.model.reset_parameters()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_args['lr'],
                                         weight_decay=self.train_args['weight_decay'])

            for epoch in range(1, self.train_args['epochs'] + 1):
                print(f"Run: {run + 1:02d}, Epoch: {epoch:02d}")
                loss = self.train(optimizer)
                hr10 = self.test()
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, HR@10: {hr10:.4f}')
                log_entry = {
                    'run': run + 1,
                    'epoch': epoch,
                    'loss': float(loss),
                    'hr10': float(hr10)
                }

                # 로그 저장
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
                
                # 최고 성능 모델 저장
                if hr10 > self.best_hr10:
                    self.best_hr10 = hr10
                    torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))

            # 각 run의 마지막 모델 저장
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, f'model_run_{run+1}.pth'))

        # 모든 학습이 끝난 후 최종 모델 저장
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'final_model.pth'))
                


    def train(self, optimizer):
        self.model.train()
        total_loss = 0
        total_samples = 0

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

        train_bar = tqdm.tqdm(loader, total=len(loader), desc="Training")
        for _, batch in enumerate(train_bar):
            batch = {k: v.to(self.train_args['device']) for k, v in batch.items()}
            optimizer.zero_grad()

            pos_edge_index = batch['pos_edge_index']
            neg_edge_index = batch['neg_edge_index']

            loss = self.model.loss(pos_edge_index, neg_edge_index)
            loss.backward()
            optimizer.step()
            
            batch_size = pos_edge_index.size(1) + neg_edge_index.size(1)  # 양성 + 음성 샘플 수
            total_loss += float(loss) * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    @torch.no_grad()
    def test(self):
        self.model.eval()

        # TODO: Implement prepare_test_data method in dataset.py

        loader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.train_args['batch_size'],
            num_workers=self.train_args['num_workers'],
            collate_fn=self.dataset.collate_fn
        )

        hits = []
        test_bar = tqdm.tqdm(loader, total=len(loader), desc="Testing")
        for _, batch in enumerate(test_bar):
            batch = {k: v.to(self.train_args['device']) for k, v in batch.items()}
            users = batch['pos_edge_index'][0].unique()
            print("unique users : ", users)
            print("batch['pos_edge_index'] : ", batch['pos_edge_index'][0])

            for user in users:
                if len(users) == 0:
                    continue # skip if no positive items
                
                # generate candidates
                user_pos_movies = batch['pos_edge_index'][1][batch['pos_edge_index'][0] == user]
                user_neg_movies = batch['neg_edge_index'][1][batch['neg_edge_index'][0] == user]
                pos_edge_index = torch.stack([user.repeat(user_pos_movies.size(0)), user_pos_movies]).to(torch.long)
                neg_edge_index = torch.stack([user.repeat(user_neg_movies.size(0)), user_neg_movies]).to(torch.long)

                if len(user_neg_movies) < self.train_args['num_neg_candidates']:
                    additional_neg_items = torch.tensor(self.dataset.find_items(user, user_neg_movies.tolist(), self.train_args['num_neg_candidates'] - len(user_neg_movies), find_similar=True), device=self.train_args['device'], dtype=torch.long)
                    neg_edge_index = torch.cat([neg_edge_index, torch.stack([user.repeat(additional_neg_items.size(0)), additional_neg_items])], dim=1)


                pos_pred = self.model.predict(pos_edge_index)
                neg_pred = self.model.predict(neg_edge_index)

                all_preds = torch.cat([pos_pred, neg_pred])
                print(f"User: {user}, Pos movies: {len(user_pos_movies)}, Neg movies: {len(user_neg_movies)}")
                print(f"Pos pred shape: {pos_pred.shape}, Neg pred shape: {neg_pred.shape}")
                print(f"All preds shape: {all_preds.shape}")
                _, indices = torch.topk(all_preds, 10)
                recommended = torch.cat([user_pos_movies, user_neg_movies])[indices]
                hit = torch.any(torch.isin(recommended, user_pos_movies)).item()
                hits.append(hit)

        return sum(hits) / len(hits)  # This is HR@10


        #         all_movies = torch.arange(self.dataset.data['movie']['num_nodes']).to(self.train_args['device'])
        #         user_tensor = user.repeat(all_movies.size(0))
        #         pred = self.model.predict(torch.stack([user_tensor, all_movies]))

        #         _, indices = torch.sort(pred, descending=True)
        #         recommended = indices[:10]

        #         hit = any(item in user_pos_movies for item in recommended)
        #         hits.append(hit)

        # return sum(hits) / len(hits)  # This is HR@10
