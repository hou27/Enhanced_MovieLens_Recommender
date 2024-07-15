import json
import os
import pickle
import torch
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from annoy import AnnoyIndex
from tqdm import tqdm

class MovieLens(torch.utils.data.Dataset):
    def __init__(self, root, name, num_core, num_negative_samples=4, num_neg_candidates=99, type=None):
        self.root = root
        self.name = name
        self.num_negative_samples = num_negative_samples
        self.num_neg_candidates = num_neg_candidates
        self.data = self.process()

        self.build_ann_index()
        self.prepare_data()

        super().__init__()

    def process(self):
        print('Processing MovieLens dataset...')
        # 데이터 로드
        movies_df = pd.read_csv(f'{self.root}/movies.csv', sep=';')
        ratings_df = pd.read_csv(f'{self.root}/ratings.csv', sep=';')
        tagging_df = pd.read_csv(f'{self.root}/tagging.csv', sep=';')
        genome_tagging_df = pd.read_csv(f'{self.root}/genome_tagging.csv', sep=';')
        tags_df = pd.read_csv(f'{self.root}/tags.csv', sep=';')
        # full_genome_tagging.csv 로드 (영화 유사도 판단용)
        full_genome_tagging_df = pd.read_csv(f'{self.root}/full_genome_tagging.csv', sep=';')

        # 딕셔너리로 데이터 구성
        data_dict = {}

        # 노드 타입별 오프셋 계산
        node_offsets = {}
        current_offset = 0
        for node_type in ['user', 'movie', 'year', 'genre', 'tag', 'genome_tag']:
            node_offsets[node_type] = current_offset
            if node_type == 'user':
                current_offset += ratings_df['uid'].max() + 1
            elif node_type == 'movie':
                current_offset += len(movies_df)
            elif node_type == 'year':
                current_offset += movies_df['year'].nunique()
            elif node_type == 'genre':
                current_offset += len(movies_df.columns[3:-3])
            elif node_type == 'tag':
                current_offset += len(tags_df)
            elif node_type == 'genome_tag':
                current_offset += genome_tagging_df['genome_tid'].nunique()

        # 영화별 특징 벡터 생성
        movie_feature_vectors = os.path.join(self.root, f'movie_feature_vectors_{self.name}.pkl')
        if os.path.exists(movie_feature_vectors):
            print('Loading existing movie feature vectors...')
            with open(movie_feature_vectors, 'rb') as f:
                self.movie_feature_vectors = pickle.load(f)
        else:
            self.movie_feature_vectors = {}
            for iid in tqdm(movies_df['iid'].unique(), desc="Generating movie feature vectors"):
            # for iid in movies_df['iid'].unique():
                feature_vector = full_genome_tagging_df[full_genome_tagging_df['iid'] == iid]['relevance'].values
                self.movie_feature_vectors[iid + node_offsets['movie']] = torch.tensor(feature_vector, dtype=torch.float)
            with open(movie_feature_vectors, 'wb') as f:
                pickle.dump(self.movie_feature_vectors, f)

        # 노드 설정
        data_dict['movie'] = {
            'num_nodes': len(movies_df)
        }
        data_dict['user'] = {
            'num_nodes': ratings_df['uid'].max() + 1
        }
        data_dict['year'] = {
            'num_nodes': movies_df['year'].nunique()
        }
        data_dict['genre'] = {
            'num_nodes': len(movies_df.columns[3:-3])
        }
        data_dict['tag'] = {
            'num_nodes': len(tags_df)
        }
        data_dict['genome_tag'] = {
            'num_nodes': genome_tagging_df['genome_tid'].nunique()
        }

        # 전체 노드 수 계산
        total_nodes = sum(data_dict[node_type]['num_nodes'] for node_type in ['user', 'movie', 'year', 'genre', 'tag', 'genome_tag'])
        data_dict['num_nodes'] = total_nodes

        # 엣지 설정 (오프셋 적용)
        user_offset = node_offsets['user']
        movie_offset = node_offsets['movie']
        user_movie_edge_index = torch.tensor(np.array([
            ratings_df['uid'].values + user_offset,
            ratings_df['iid'].values + movie_offset
        ]), dtype=torch.long)
        data_dict[('user', 'rates', 'movie')] = {
            'edge_index': user_movie_edge_index,
            'edge_attr': torch.tensor(ratings_df['rating'].values, dtype=torch.float)
        }

        year_offset = node_offsets['year']
        year_mapping = {year: idx + year_offset for idx, year in enumerate(movies_df['year'].unique())}
        movie_year_edge_index = torch.tensor(np.array([
            movies_df.index.values + movie_offset,
            [year_mapping[year] for year in movies_df['year'].values]
        ]), dtype=torch.long)
        data_dict[('movie', 'in', 'year')] = {
            'edge_index': movie_year_edge_index
        }

        genre_offset = node_offsets['genre']
        genre_edges = []
        for idx, row in movies_df.iterrows():
            for genre_idx, is_genre in enumerate(row[3:-3]):
                if is_genre:
                    genre_edges.append((idx + movie_offset, genre_idx + genre_offset))
        movie_genre_edge_index = torch.tensor(list(zip(*genre_edges)), dtype=torch.long)
        data_dict[('movie', 'of', 'genre')] = {
            'edge_index': movie_genre_edge_index
        }

        tag_offset = node_offsets['tag']
        tag_mapping = {tid: idx + tag_offset for idx, tid in enumerate(tags_df['tid'])}
        tagging_tid = tagging_df['tid'].values
        tagging_iid = tagging_df['iid'].values
        tag2item_edge_index = torch.tensor(np.array([
            [tag_mapping[tid] for tid in tagging_tid],
            tagging_iid + movie_offset
        ]), dtype=torch.long)
        data_dict[('tag', 'describes', 'movie')] = {
            'edge_index': tag2item_edge_index
        }

        tag2user_edge_index = torch.tensor(np.array([
            [tag_mapping[tid] for tid in tagging_tid],
            tagging_df['uid'].values + user_offset
        ]), dtype=torch.long)
        data_dict[('tag', 'used_by', 'user')] = {
            'edge_index': tag2user_edge_index
        }

        genome_tag_offset = node_offsets['genome_tag']
        genome_tag_mapping = {tid: idx + genome_tag_offset for idx, tid in enumerate(genome_tagging_df['genome_tid'].unique())}
        genome_tag_tid = genome_tagging_df['genome_tid'].values
        genome_tag_iid = genome_tagging_df['iid'].values
        genome_tag2item_edge_index = torch.tensor(np.array([
            [genome_tag_mapping[tid] for tid in genome_tag_tid],
            genome_tag_iid + movie_offset
        ]), dtype=torch.long)
        data_dict[('genome_tag', 'describes', 'movie')] = {
            'edge_index': genome_tag2item_edge_index
        }

        # train/test 분할 (오프셋 적용)
        user_movie_pairs = ratings_df[['uid', 'iid', 'rating']].values
        user_movie_pairs[:, 0] += user_offset
        user_movie_pairs[:, 1] += movie_offset
        train_pairs, test_pairs = train_test_split(user_movie_pairs, test_size=0.2, random_state=42)

        # positive/negative 샘플 생성
        train_pos = train_pairs[train_pairs[:, 2] >= 3]
        train_neg = train_pairs[train_pairs[:, 2] < 3]
        test_pos = test_pairs[test_pairs[:, 2] >= 3]
        test_neg = test_pairs[test_pairs[:, 2] < 3]

        data_dict['train_pos'] = torch.tensor(np.array(train_pos[:, :2]), dtype=torch.long)
        data_dict['train_neg'] = torch.tensor(np.array(train_neg[:, :2]), dtype=torch.long)
        data_dict['test_pos'] = torch.tensor(np.array(test_pos[:, :2]), dtype=torch.long)
        data_dict['test_neg'] = torch.tensor(np.array(test_neg[:, :2]), dtype=torch.long)

        data_dict['node_types'] = ['user', 'movie', 'year', 'genre', 'tag', 'genome_tag']
        data_dict['edge_types'] = [
            ('user', 'rates', 'movie'),
            ('movie', 'in', 'year'),
            ('movie', 'of', 'genre'),
            ('tag', 'describes', 'movie'),
            ('tag', 'used_by', 'user'),
            ('genome_tag', 'describes', 'movie')
        ]

        # 저장
        with open(f'{self.root}/ml_25m_core_10_type_hete.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

        return data_dict
    
    def build_ann_index(self):
        print('Building Annoy index...')
        ann_index_file = os.path.join(self.root, f'ann_index_{self.name}_euclidean.ann')
        ann_mapping_file = os.path.join(self.root, f'ann_mapping_{self.name}_euclidean.pkl')

        if os.path.exists(ann_index_file) and os.path.exists(ann_mapping_file):
            print('Loading existing Annoy index...')
            self.ann_index = AnnoyIndex(1128, 'euclidean')  # 'angular'에서 'euclidean'으로 변경
            self.ann_index.load(ann_index_file)
            with open(ann_mapping_file, 'rb') as f:
                mappings = pickle.load(f)
            self.item_to_ann_index = mappings['item_to_ann_index']
            self.ann_index_to_item = mappings['ann_index_to_item']
        else:
            print('Building new Annoy index...')
            self.ann_index = AnnoyIndex(1128, 'euclidean')  # 'angular'에서 'euclidean'으로 변경
            self.item_to_ann_index = {}
            self.ann_index_to_item = {}
            ann_index = 0

            for item, vec in tqdm(self.movie_feature_vectors.items(), desc="Building ANN index"):
                if vec.shape[0] == 0:
                    continue
                self.ann_index.add_item(ann_index, vec.numpy())
                self.item_to_ann_index[item] = ann_index
                self.ann_index_to_item[ann_index] = item
                ann_index += 1

            self.ann_index.build(10)
            self.ann_index.save(ann_index_file)
            mappings = {
                'item_to_ann_index': self.item_to_ann_index,
                'ann_index_to_item': self.ann_index_to_item
            }
            with open(ann_mapping_file, 'wb') as f:
                pickle.dump(mappings, f)

    def prepare_data(self):
        train_data_file = os.path.join(self.root, f'train_data_{self.name}.pkl')
        test_data_file = os.path.join(self.root, f'test_data_{self.name}.pkl')
        user_pos_items_file = os.path.join(self.root, f'user_pos_items_{self.name}.pkl')
        user_neg_items_file = os.path.join(self.root, f'user_neg_items_{self.name}.pkl')

        if os.path.exists(train_data_file) and os.path.exists(test_data_file) and os.path.exists(user_pos_items_file) and os.path.exists(user_neg_items_file):
            print('Loading existing train and test data...')
            with open(train_data_file, 'rb') as f:
                self.train_data = pickle.load(f)
            with open(test_data_file, 'rb') as f:
                self.test_data = pickle.load(f)
            with open(user_pos_items_file, 'rb') as f:
                self.user_pos_items = pickle.load(f)
            with open(user_neg_items_file, 'rb') as f:
                self.user_neg_items = pickle.load(f)
        else:
            print('Preparing train and test data...')
            self.prepare_train_test_data()
            
            try:
                with open(train_data_file, 'wb') as f:
                    pickle.dump(self.train_data, f)
                with open(test_data_file, 'wb') as f:
                    pickle.dump(self.test_data, f)
                with open(user_pos_items_file, 'wb') as f:
                    pickle.dump(self.user_pos_items, f)
                with open(user_neg_items_file, 'wb') as f:
                    pickle.dump(self.user_neg_items, f)
                print('Data successfully saved.')
            except Exception as e:
                print(f'Error saving data: {e}')
            

    def prepare_train_test_data(self):
        log_file = os.path.join('logs', 'prepare_train_test_data.jsonl')
        self.train_data = []
        self.test_data = {}
        self.user_pos_items = {}
        self.user_neg_items = {}

        user_movie_edge_index = self.data[('user', 'rates', 'movie')]['edge_index']
        user_movie_ratings = self.data[('user', 'rates', 'movie')]['edge_attr']

        # all_items = set(self.movie_feature_vectors.keys())

        for user in tqdm(torch.unique(user_movie_edge_index[0]), desc="Preparing train and test data"):
            user = user.item()
            user_items = user_movie_edge_index[1][user_movie_edge_index[0] == user]
            user_ratings = user_movie_ratings[user_movie_edge_index[0] == user]

            pos_items = user_items[user_ratings >= 3].tolist()
            neg_items = user_items[user_ratings < 3].tolist()

            self.user_pos_items[user] = pos_items
            self.user_neg_items[user] = neg_items

            # 테스트용 양성 샘플 선택
            if len(pos_items) == 0:
                print(f"User {user} has no positive samples")
                continue
            test_pos_item = pos_items.pop()

            train_neg_items = set()
            user_interacted_items = set(pos_items + neg_items + [test_pos_item])
            
            # print(f"User {user}: {len(pos_items)} positive samples, {len(neg_items)} negative samples")
            for pos_item in pos_items:
                # remaining_neg_items = list(set(neg_items) - train_neg_items)  # 중복 제거를 위해 set 사용
                neg_samples_for_pos = neg_items.copy() # random.sample(remaining_neg_items, min(len(remaining_neg_items), self.num_negative_samples))

                # 추가 음성 샘플이 필요한 경우
                while len(neg_samples_for_pos) < self.num_negative_samples:
                    # candidates = list(user_interacted_items - train_neg_items - set(neg_samples_for_pos))
                    candidates = list(user_interacted_items - set(neg_samples_for_pos))
                    if not candidates:
                        print(f"Warning: No more candidates for user {user}")
                        break

                    # 추가 음성 샘플 찾기
                    additional_neg_items = self.find_items(user, [pos_item], self.num_negative_samples - len(neg_samples_for_pos), find_similar=False)
                    # additional_neg_items = [item for item in additional_neg_items if item not in user_interacted_items and item not in train_neg_items]
                    additional_neg_items = [item for item in additional_neg_items if item not in user_interacted_items]

                    if not additional_neg_items and len(neg_items) > 0:
                        # print(f"User {user}: No more negative samples for positive item {pos_item}")
                        # print(f"Finding similar items for negative samples with {len(neg_items)} items")
                        additional_neg_items = self.find_items(user, neg_items, (self.num_negative_samples - len(neg_samples_for_pos)) + 10, find_similar=True)
                        additional_neg_items = [item for item in additional_neg_items if item not in user_interacted_items]
                        # print(f"User {user}: Found {len(additional_neg_items)} similar items for negative samples")

                    if additional_neg_items:
                        neg_samples_for_pos.extend(additional_neg_items[:self.num_negative_samples - len(neg_samples_for_pos)])
                    else:
                        # print("No more negative samples for user")
                        # print(f"Negative samples for positive item {pos_item}: {neg_samples_for_pos}")
                        additional_neg_items = self.find_items(user, pos_items, (self.num_negative_samples - len(neg_samples_for_pos)) + 10, find_similar=False)
                        # print(f"User {user}: Found similar items : {additional_neg_items} for negative samples")
                        additional_neg_items = [item for item in additional_neg_items if item not in user_interacted_items]
                        # print(f"User {user}: Found {len(additional_neg_items)} similar items for negative samples")
                        neg_samples_for_pos.extend(additional_neg_items[:self.num_negative_samples - len(neg_samples_for_pos)])

                neg_samples_for_current_pos = random.sample(neg_samples_for_pos, self.num_negative_samples)
                for neg_item in neg_samples_for_current_pos:
                    self.train_data.append((user, pos_item, neg_item))
                    train_neg_items.add(neg_item)

            # print(f"User {user}: {len(train_neg_items)} negative samples for positive items")
            # 테스트 데이터 준비
            available_neg_items = list(set(neg_items) - train_neg_items)

            # 필요한 추가 음성 샘플 수 계산
            num_additional_needed = max(0, self.num_neg_candidates - len(available_neg_items))

            while len(available_neg_items) < self.num_neg_candidates:
                # print(f"User {user}: Finding {num_additional_needed} additional negative samples")
                additional_items = self.find_items(user, [test_pos_item], num_additional_needed, find_similar=False)
                
                # 새로운 아이템만 추가
                new_items = [item for item in additional_items 
                            if item not in user_interacted_items 
                            and item not in train_neg_items 
                            and item not in available_neg_items]
                
                available_neg_items.extend(new_items)
                
                # 아직 99개를 채우지 못했다면 다시
                num_additional_needed = self.num_neg_candidates - len(available_neg_items)
                
                if not new_items:
                    # print(f"Warning: User {user}: Could not find new negative samples. Trying with different strategy.")
                    # 다른 전략 시도
                    additional_items = self.find_items(user, neg_items, num_additional_needed + 10, find_similar=True)
                    new_items = [item for item in additional_items 
                                if item not in user_interacted_items 
                                and item not in train_neg_items 
                                and item not in available_neg_items]
                    available_neg_items.extend(new_items)
                    
                    if not new_items:
                        # print(f"Error: User {user}: Failed to find enough negative samples. Current count: {len(available_neg_items)}")
                        break  # 무한 루프 방지

            # print(f"User {user}: Selecting 99(num_neg_candidates) negative samples from {len(available_neg_items)} available neg items")
            test_neg_items = random.sample(available_neg_items, min(self.num_neg_candidates, len(available_neg_items)))

            self.test_data[user] = {
                'pos_item': test_pos_item,
                'neg_items': test_neg_items
            }
            log_entry = {
                'user': user,
                'pos_item': test_pos_item,
                'neg_items': len(test_neg_items),
                # 'test_items': f"{test_neg_items}",
                # 'interacted_items': f"{user_interacted_items}",
            }
            # 로그 저장
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')


    def find_items(self, user, items, n, find_similar=True):
        user = user.item() if isinstance(user, torch.Tensor) else user
        items = [item.item() if isinstance(item, torch.Tensor) else item for item in items]
        
        all_items = set(self.movie_feature_vectors.keys())
        user_items = set(self.user_pos_items[user] + self.user_neg_items[user])
        candidate_items = list(all_items - user_items)
        
        item_scores = []
        for item in items:
            if item not in self.item_to_ann_index:
                continue
            item_idx = self.item_to_ann_index[item]
            similar_items = self.ann_index.get_nns_by_item(item_idx, n + len(user_items))
            
            for sim_idx in similar_items:
                candidate = self.ann_index_to_item[sim_idx]
                if candidate in candidate_items:
                    distance = self.ann_index.get_distance(item_idx, sim_idx)
                    score = 1 / (1 + distance) if find_similar else distance  # 유클리디안 거리에 맞게 스코어 계산 방식 변경
                    item_scores.append((candidate, score))
        
        item_scores.sort(key=lambda x: x[1], reverse=find_similar)
        return [item for item, _ in item_scores[:n]]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        user, pos_item, neg_item = self.train_data[idx]
        return {'user': user, 'pos_item': pos_item, 'neg_item': neg_item}

    def get_test_samples(self, user):
        if user not in self.test_data:
            return None
        
        pos_item = self.test_data[user]['pos_item']
        neg_items = self.test_data[user]['neg_items']
        
        test_items = [pos_item] + neg_items
        labels = torch.zeros(len(test_items), dtype=torch.long)
        labels[0] = 1  # 첫 번째 아이템이 양성 샘플
        
        return torch.tensor([user] * len(test_items)), torch.tensor(test_items), labels

    def collate_fn(self, batch):
        users = torch.tensor([item['user'] for item in batch])
        pos_items = torch.tensor([item['pos_item'] for item in batch])
        neg_items = torch.tensor([item['neg_item'] for item in batch])
        return {
            'users': users,
            'pos_items': pos_items,
            'neg_items': neg_items
        }