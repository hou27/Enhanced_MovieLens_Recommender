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
        for batch in train_bar:
            users = batch['users'].to(self.train_args['device'])
            pos_items = batch['pos_items'].to(self.train_args['device'])
            neg_items = batch['neg_items'].to(self.train_args['device'])

            pos_edge_index = torch.stack([users, pos_items])
            neg_edge_index = torch.stack([users, neg_items])

            loss = self.model.loss(pos_edge_index, neg_edge_index)
            loss.backward()
            optimizer.step()
            
            batch_size = users.size(0)  # 양성 + 음성 샘플 수
            total_loss += float(loss) * batch_size
            total_samples += batch_size

            train_bar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

        return total_loss / total_samples

    @torch.no_grad()
    def test(self):
        self.model.eval()

        hits = []
        all_users = list(self.dataset.test_data.keys())
        test_bar = tqdm.tqdm(all_users, total=len(all_users), desc="Testing")
        
        for user in test_bar:
            test_samples = self.dataset.get_test_samples(user)
            if test_samples is None:
                continue

            users, items, labels = test_samples
            users = users.to(self.train_args['device'])
            items = items.to(self.train_args['device'])
            labels = labels.to(self.train_args['device'])

            edge_index = torch.stack([users, items])
            predictions = self.model.predict(edge_index)
            
            _, indices = torch.topk(predictions, 10)
            hit = torch.any(labels[indices] == 1).item()
            hits.append(hit)

            test_bar.set_postfix(HR10=f"{sum(hits) / len(hits):.4f}")

        return sum(hits) / len(hits)  # This is HR@10
