import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


class MovieLens(torch.utils.data.Dataset):
    def __init__(self, root, name, num_core, num_negative_samples=4, num_neg_candidates=99, type=None):
        self.root = root
        self.name = name
        self.type = type
        self.num_negative_samples = num_negative_samples
        self.num_neg_candidates = num_neg_candidates
        self.data = self.process()

        self.prepare_train_test_data()

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
        ratings_df = ratings_df.sort_values(['uid', 'timestamp'])
    
        train_ratings = []
        test_ratings = []
        
        for uid, group in ratings_df.groupby('uid'):
            user_ratings = group.values
            if len(user_ratings) > 1:
                train_ratings.extend(user_ratings[:-1])
                test_ratings.append(user_ratings[-1])
            else:
                train_ratings.extend(user_ratings)
        
        train_ratings = np.array(train_ratings)
        test_ratings = np.array(test_ratings)
        
        # positive/negative 샘플 생성
        train_pos = train_ratings[train_ratings[:, 2] >= 4]
        train_neg = train_ratings[train_ratings[:, 2] < 4]
        test_pos = test_ratings[test_ratings[:, 2] >= 4]
        test_neg = test_ratings[test_ratings[:, 2] < 4]

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

        data_dict['ratings'] = ratings_df

        # 저장
        with open(f'{self.root}/ml_25m_core_10_type_hete.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

        return data_dict

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
        print('Creating rating property edges...')
        test_pos_unid_inid_map, test_neg_unid_inid_map, neg_unid_inid_map = {}, {}, {}
        train_pos_unid_inid_map = {}

        rating_np = np.zeros((0,))
        user2item_edge_index_np = np.zeros((2, 0))
        sorted_ratings = self.data['ratings']
        all_items = set(sorted_ratings.iid.unique())

        pbar = tqdm(sorted_ratings.uid.unique(), total=len(sorted_ratings.uid.unique()))
        for uid in pbar:
            pbar.set_description(f'Creating the edges for the user {uid}')
            uid_ratings = sorted_ratings[sorted_ratings.uid == uid]
            uid_iids = uid_ratings.iid.to_numpy()
            uid_ratings_values = uid_ratings.rating.to_numpy()

            # 별점 4점 이상인 아이템을 positive로 간주
            pos_items = uid_iids[uid_ratings_values >= 4]
            # # 별점 4점 미만인 아이템을 negative로 간주
            # explicit_neg_items = uid_iids[uid_ratings_values < 4]
            # 별점 2점 이하인 아이템을 negative로 간주
            explicit_neg_items = uid_iids[uid_ratings_values <= 2]
            
            # Unobserved 아이템들 (사용자가 평가하지 않은 모든 아이템)
            unobserved_items = list(all_items - set(uid_iids))
            
            if len(pos_items) > 0:
                test_pos_uid_iids = [pos_items[-1]]  # 가장 마지막 positive 아이템을 테스트 셋으로
                train_pos_uid_iids = list(set(pos_items) - set(test_pos_uid_iids))
                
                test_pos_unid_inid_map[uid] = test_pos_uid_iids
                train_pos_unid_inid_map[uid] = train_pos_uid_iids

                if len(explicit_neg_items) > 0:
                    test_neg_uid_iids = [explicit_neg_items[-1]]  # 가장 마지막 negative 아이템을 테스트 셋으로
                    remaining_neg_items = list(set(explicit_neg_items) - set(test_neg_uid_iids))
                    
                    test_neg_unid_inid_map[uid] = test_neg_uid_iids
                else:
                    remaining_neg_items = []
                
                # Negative 아이템들 (4점 미만인 나머지 아이템들 + unobserved 아이템들)
                neg_uid_iids = remaining_neg_items + unobserved_items
                neg_unid_inid_map[uid] = neg_uid_iids

                unid_user2item_edge_index_np = np.array(
                    [[uid for _ in range(len(train_pos_uid_iids))], train_pos_uid_iids]
                )
                user2item_edge_index_np = np.hstack([user2item_edge_index_np, unid_user2item_edge_index_np])

                train_pos_uid_ratings = uid_ratings_values[np.isin(uid_iids, train_pos_uid_iids)]
                rating_np = np.concatenate([rating_np, train_pos_uid_ratings])
                
        self.data['rating_np'] = rating_np
        self.data['edge_index_nps'] = {'user2item': user2item_edge_index_np}

        self.data['test_pos_unid_inid_map'] = test_pos_unid_inid_map
        self.data['test_neg_unid_inid_map'] = test_neg_unid_inid_map
        self.data['neg_unid_inid_map'] = neg_unid_inid_map
        self.data['train_pos_unid_inid_map'] = train_pos_unid_inid_map

        # print(train_pos_unid_inid_map)
    def cf_negative_sampling(self):
        print('CF negative sampling...')
        
        self.train_pos_unid_inid_map = self.data['train_pos_unid_inid_map']
        self.test_pos_unid_inid_map = self.data['test_pos_unid_inid_map']
        self.neg_unid_inid_map = self.data['neg_unid_inid_map']

        train_data = []
        for u_nid, pos_inids in tqdm(self.train_pos_unid_inid_map.items(), desc="Generating samples"):
            neg_inids = self.neg_unid_inid_map[u_nid]
            
            for pos_inid in pos_inids:
                neg_samples = np.random.choice(neg_inids, size=self.num_negative_samples, replace=False)
                for neg_inid in neg_samples:
                    train_data.append([u_nid, pos_inid, neg_inid])

        train_data_np = np.array(train_data)
        train_data_t = torch.from_numpy(train_data_np).long()
        shuffle_idx = torch.randperm(train_data_t.shape[0])
        self.train_data = train_data_t[shuffle_idx]
        self.train_data_length = train_data_t.shape[0]

    def __len__(self):
        return self.train_data_length

    def __getitem__(self, idx):        
        if not hasattr(self, 'train_data'):
            self.cf_negative_sampling()
        
        if idx >= self.train_data_length:
            raise IndexError("Index out of range")
        
        train_data_t = self.train_data[idx]
        
        return {
            'user': train_data_t[0].item(),
            'pos_item': train_data_t[1].item(),
            'neg_item': train_data_t[2].item()
        }

    def get_test_samples(self, unid):
        if unid not in self.data['test_pos_unid_inid_map']:
            return None
        
        pos_inid = self.data['test_pos_unid_inid_map'][unid][0]
        neg_inids = self.data['neg_unid_inid_map'][unid]
        
        if not neg_inids:
            return None
        
        test_inids = [pos_inid] + random.sample(neg_inids, self.num_neg_candidates)# min(len(neg_inids), self.num_neg_candidates - 1))
        labels = torch.zeros(len(test_inids), dtype=torch.long)
        labels[0] = 1  # 첫 번째 아이템이 양성 샘플
        
        return torch.tensor([unid] * len(test_inids)), torch.tensor(test_inids), labels

    def collate_fn(self, batch):
        users = torch.tensor([item['user'] for item in batch])
        pos_items = torch.tensor([item['pos_item'] for item in batch])
        neg_items = torch.tensor([item['neg_item'] for item in batch])
        return {
            'users': users,
            'pos_items': pos_items,
            'neg_items': neg_items
        }