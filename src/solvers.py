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
    def __init__(self, model_class, dataset, model_args, train_args):
        self.model_class = model_class
        self.dataset = dataset
        self.model_args = model_args
        self.train_args = train_args
        self.log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'training_log.jsonl')
        self.best_hr10 = 0

    def run(self):
        for run in range(self.train_args['runs']):
            # 시드 고정 코드 추가
            seed = 2019 + run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            device = self.train_args['device']
            self.model_args['num_nodes'] = self.dataset.data["num_nodes"]
            self.model_args['dataset'] = self.dataset
            self.model_args['train_args'] = self.train_args
            self.model = self.model_class(**self.model_args).to(device)

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
        total_loss = []

        self.dataset.cf_negative_sampling()

        loader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.train_args['batch_size'],
            num_workers=self.train_args['num_workers'],
            collate_fn=self.dataset.collate_fn
        )

        train_bar = tqdm.tqdm(loader, total=len(loader), desc="Training")
        for batch in train_bar:
            users = batch['users'].to(self.train_args['device'])
            pos_items = batch['pos_items'].to(self.train_args['device'])
            neg_items = batch['neg_items'].to(self.train_args['device'])

            pos_edge_index = torch.stack([users, pos_items])
            neg_edge_index = torch.stack([users, neg_items])

            optimizer.zero_grad() # 각 배치마다 새로운 그래디언트 계산
            # loss 계산
            loss = self.model.loss(pos_edge_index, neg_edge_index)
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트
            
            total_loss.append(loss.detach().cpu().item())
            train_loss = np.mean(total_loss)

            train_bar.set_postfix(loss=f"{train_loss:.4f}")

        return train_loss
    
    @torch.no_grad()
    def test(self):
        self.model.eval()

        hits = []
        all_users = list(self.dataset.data['test_pos_unid_inid_map'].keys())
        test_bar = tqdm.tqdm(all_users, total=len(all_users), desc="Testing")
        
        for unid in test_bar:
            test_samples = self.dataset.get_test_samples(unid)
            if test_samples is None:
                continue

            users, items, labels = test_samples
            if len(users) < 30:
                continue
            users = users.to(self.train_args['device'])
            items = items.to(self.train_args['device'])
            labels = labels.to(self.train_args['device'])

            edge_index = torch.stack([users, items])
            predictions = self.model.predict(edge_index)
            
            _, indices = torch.topk(predictions, 10)
            hit = torch.any(labels[indices] == 1).item()
            hits.append(hit)

            test_bar.set_postfix(HR10=f"{sum(hits) / len(hits):.4f}")

        return sum(hits) / len(hits)  # HR@10