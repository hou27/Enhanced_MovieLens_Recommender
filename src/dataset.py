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
    def __init__(self, root, name, num_core, num_negative_samples=4, type=None):
        self.root = root
        self.name = name
        self.num_negative_samples = num_negative_samples
        self.data = self.process()

        self.build_ann_index()
        self.prepare_train_data()

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
        self.movie_feature_vectors = {}
        for iid in tqdm(movies_df['iid'].unique(), desc="Generating movie feature vectors"):
        # for iid in movies_df['iid'].unique():
            feature_vector = full_genome_tagging_df[full_genome_tagging_df['iid'] == iid]['relevance'].values
            self.movie_feature_vectors[iid + node_offsets['movie']] = torch.tensor(feature_vector, dtype=torch.float)

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
    
    def prepare_train_data(self):
        print('Preparing training data...')
        user_movie_edge_index = self.data[('user', 'rates', 'movie')]['edge_index']
        user_movie_ratings = self.data[('user', 'rates', 'movie')]['edge_attr']

        self.pos_edges = user_movie_edge_index[:, user_movie_ratings >= 3]
        self.neg_edges = user_movie_edge_index[:, user_movie_ratings < 3]

        self.user_pos_items = {}
        self.user_neg_items = {}
        for user in tqdm(torch.unique(user_movie_edge_index[0]), desc="Dividing user items"):
        # for user in torch.unique(user_movie_edge_index[0]):
            user_items = user_movie_edge_index[1][user_movie_edge_index[0] == user]
            user_ratings = user_movie_ratings[user_movie_edge_index[0] == user]
            
            self.user_pos_items[user.item()] = user_items[user_ratings >= 3].tolist()
            self.user_neg_items[user.item()] = user_items[user_ratings < 3].tolist()

        self.min_samples = 10  # 최소 샘플 수 설정
        self.augment_samples()
        self.prepare_train_pairs()

    def augment_samples(self):
        for user in tqdm(self.user_pos_items.keys(), desc="Augmenting samples"):
        # for user in self.user_pos_items.keys():
            if len(self.user_pos_items[user]) < self.min_samples:
                self.user_pos_items[user].extend(self.find_items(user, self.user_pos_items[user], self.min_samples - len(self.user_pos_items[user]), find_similar=True))
            
            if len(self.user_neg_items[user]) < self.min_samples:
                if len(self.user_neg_items[user]) == 0:
                    self.user_neg_items[user] = self.find_items(user, self.user_pos_items[user], self.min_samples, find_similar=False)
                else:
                    self.user_neg_items[user].extend(self.find_items(user, self.user_neg_items[user], self.min_samples - len(self.user_neg_items[user]), find_similar=True))
    
    # Annoy 라이브러리를 사용하여 유사한 아이템 찾기
    def build_ann_index(self):
        print('Building Annoy index...')
        ann_index_file = os.path.join(self.root, f'ann_index_{self.name}.ann')
        ann_mapping_file = os.path.join(self.root, f'ann_mapping_{self.name}.pkl')

        if os.path.exists(ann_index_file) and os.path.exists(ann_mapping_file):
            print('Loading existing Annoy index...')
            self.ann_index = AnnoyIndex(1128, 'angular')
            self.ann_index.load(ann_index_file)
            with open(ann_mapping_file, 'rb') as f:
                mappings = pickle.load(f)
            self.item_to_ann_index = mappings['item_to_ann_index']
            self.ann_index_to_item = mappings['ann_index_to_item']
        else:
            print('Building new Annoy index...')
            self.ann_index = AnnoyIndex(1128, 'angular')
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

    def find_items(self, user, items, n, find_similar=True):
        user = user.item() if isinstance(user, torch.Tensor) else user
        items = [item.item() if isinstance(item, torch.Tensor) else item for item in items]
        
        all_items = set(self.movie_feature_vectors.keys())
        user_items = set(self.user_pos_items[user] + self.user_neg_items[user])
        candidate_items = list(all_items - user_items)
        
        item_scores = []
        for item in items:
            if item not in self.item_to_ann_index:
                continue  # Skip items with 0-dimension feature vector

            item_idx = self.item_to_ann_index[item]
            similar_items = self.ann_index.get_nns_by_item(item_idx, n + len(user_items))
            
            for sim_idx in similar_items:
                candidate = self.ann_index_to_item[sim_idx]
                if candidate in candidate_items:
                    similarity = 1 - self.ann_index.get_distance(item_idx, sim_idx) / 2  # convert angular distance to similarity
                    score = similarity if find_similar else 1 - similarity
                    item_scores.append((candidate, score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in item_scores[:n]]

    def prepare_train_pairs(self):
        print('Preparing training pairs...')
        self.train_pairs = []
        for user in tqdm(self.user_pos_items.keys(), desc="Preparing training pairs"):
        # for user in self.user_pos_items.keys():
            pos_items = self.user_pos_items[user]
            neg_items = self.user_neg_items[user]
            for pos_item in pos_items:
                for _ in range(self.num_negative_samples):
                    neg_item = random.choice(neg_items)
                    self.train_pairs.append((user, pos_item, neg_item))

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        user, pos_item, neg_item = self.train_pairs[idx]
        pos_edge = torch.tensor([user, pos_item])
        neg_edge = torch.tensor([user, neg_item])
        return {'pos_edge': pos_edge, 'neg_edge': neg_edge}

    def collate_fn(self, batch):
        pos_edges = torch.stack([item['pos_edge'] for item in batch], dim=1)
        neg_edges = torch.stack([item['neg_edge'] for item in batch], dim=1)
        return {
            'pos_edge_index': pos_edges,
            'neg_edge_index': neg_edges
        }