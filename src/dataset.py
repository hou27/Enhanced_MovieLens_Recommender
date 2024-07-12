import pickle
import torch
# from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MovieLens(torch.utils.data.Dataset):
    def __init__(self, root, name, num_core, pos_neg_ratio = 0.2, type=None, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.data = self.process()

        # positive/negative 샘플 생성
        self.pos_neg_ratio = pos_neg_ratio
        self.prepare_train_data()

        super().__init__()

    def process(self):
        # 데이터 로드
        movies_df = pd.read_csv(f'{self.root}/movies.csv', sep=';')
        ratings_df = pd.read_csv(f'{self.root}/ratings.csv', sep=';')
        tagging_df = pd.read_csv(f'{self.root}/tagging.csv', sep=';')
        genome_tagging_df = pd.read_csv(f'{self.root}/genome_tagging.csv', sep=';')
        tags_df = pd.read_csv(f'{self.root}/tags.csv', sep=';')

        # # HeteroData 객체 생성
        # data = HeteroData()

        # # 노드 타입별 오프셋 계산
        # self.node_offsets = {}
        # current_offset = 0
        # for node_type in ['user', 'movie', 'year', 'genre', 'tag', 'genome_tag']:
        #     self.node_offsets[node_type] = current_offset
        #     if node_type == 'user':
        #         current_offset += ratings_df['uid'].max() + 1
        #     elif node_type == 'movie':
        #         current_offset += len(movies_df)
        #     elif node_type == 'year':
        #         current_offset += movies_df['year'].nunique()
        #     elif node_type == 'genre':
        #         current_offset += len(movies_df.columns[3:-3])
        #     elif node_type == 'tag':
        #         current_offset += len(tags_df)
        #     elif node_type == 'genome_tag':
        #         current_offset += genome_tagging_df['genome_tid'].nunique()

        # # 노드 설정
        # data['movie'].num_nodes = len(movies_df)
        # data['user'].num_nodes = ratings_df['uid'].max() + 1
        # data['year'].num_nodes = movies_df['year'].nunique()
        # data['genre'].num_nodes = len(movies_df.columns[3:-3])
        # data['tag'].num_nodes = len(tags_df)
        # data['genome_tag'].num_nodes = genome_tagging_df['genome_tid'].nunique()

        # # 엣지 설정 (오프셋 적용)
        # user_offset = self.node_offsets['user']
        # movie_offset = self.node_offsets['movie']
        # user_movie_edge_index = torch.tensor(np.array([
        #     ratings_df['uid'].values + user_offset,
        #     ratings_df['iid'].values + movie_offset
        # ]), dtype=torch.long)
        # data['user', 'rates', 'movie'].edge_index = user_movie_edge_index
        # data['user', 'rates', 'movie'].edge_attr = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

        # year_offset = self.node_offsets['year']
        # year_mapping = {year: idx + year_offset for idx, year in enumerate(movies_df['year'].unique())}
        # data['movie', 'in', 'year'].edge_index = torch.tensor(np.array([
        #     movies_df.index.values + movie_offset,
        #     [year_mapping[year] for year in movies_df['year'].values]
        # ]), dtype=torch.long)

        # genre_offset = self.node_offsets['genre']
        # genre_edges = []
        # for idx, row in movies_df.iterrows():
        #     for genre_idx, is_genre in enumerate(row[3:-3]):
        #         if is_genre:
        #             genre_edges.append((idx + movie_offset, genre_idx + genre_offset))
        # data['movie', 'of', 'genre'].edge_index = torch.tensor(list(zip(*genre_edges)), dtype=torch.long)

        # tag_offset = self.node_offsets['tag']
        # tag_mapping = {tid: idx + tag_offset for idx, tid in enumerate(tags_df['tid'])}
        # tag2item_edges = [
        #     [tag_mapping[tid] for tid in tagging_df['tid']],
        #     tagging_df['iid'].values + movie_offset
        # ]
        # data['tag', 'describes', 'movie'].edge_index = torch.tensor(np.array(tag2item_edges), dtype=torch.long)

        # tag2user_edges = [
        #     [tag_mapping[tid] for tid in tagging_df['tid']],
        #     tagging_df['uid'].values + user_offset
        # ]
        # data['tag', 'used_by', 'user'].edge_index = torch.tensor(np.array(tag2user_edges), dtype=torch.long)

        # genome_tag_offset = self.node_offsets['genome_tag']
        # genome_tag_mapping = {tid: idx + genome_tag_offset for idx, tid in enumerate(genome_tagging_df['genome_tid'].unique())}
        # genome_tag2item_edges = [
        #     [genome_tag_mapping[tid] for tid in genome_tagging_df['genome_tid']],
        #     genome_tagging_df['iid'].values + movie_offset
        # ]
        # data['genome_tag', 'describes', 'movie'].edge_index = torch.tensor(np.array(genome_tag2item_edges), dtype=torch.long)

        # # train/test 분할 (오프셋 적용)
        # user_movie_pairs = ratings_df[['uid', 'iid', 'rating']].values
        # user_movie_pairs[:, 0] += user_offset
        # user_movie_pairs[:, 1] += movie_offset
        # train_pairs, test_pairs = train_test_split(user_movie_pairs, test_size=0.2, random_state=42)

        # # positive/negative 샘플 생성
        # train_pos = train_pairs[train_pairs[:, 2] >= 3]
        # train_neg = train_pairs[train_pairs[:, 2] < 3]
        # test_pos = test_pairs[test_pairs[:, 2] >= 3]
        # test_neg = test_pairs[test_pairs[:, 2] < 3]

        # self.train_pos = torch.tensor(np.array(train_pos[:, :2]), dtype=torch.long)
        # self.train_neg = torch.tensor(np.array(train_neg[:, :2]), dtype=torch.long)
        # self.test_pos = torch.tensor(np.array(test_pos[:, :2]), dtype=torch.long)
        # self.test_neg = torch.tensor(np.array(test_neg[:, :2]), dtype=torch.long)

        # return data
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

    # def __len__(self):
    #     return len(self.data['train_pos']) + len(self.data['train_neg'])

    # def __len__(self):
    #     return len(self.data[('user', 'rates', 'movie')]['edge_index'][0])

    # def __getitem__(self, idx):
    #     user = self.data[('user', 'rates', 'movie')]['edge_index'][0][idx]
    #     movie = self.data[('user', 'rates', 'movie')]['edge_index'][1][idx]
    #     rating = self.data[('user', 'rates', 'movie')]['edge_attr'][idx]

    #     return {
    #         'user': user,
    #         'movie': movie,
    #         'rating': rating
    #     }

    # def collate_fn(self, batch):
    #     users = torch.tensor([item['user'] for item in batch])
    #     movies = torch.tensor([item['movie'] for item in batch])
    #     ratings = torch.tensor([item['rating'] for item in batch])

    #     return {
    #         ('user', 'rates', 'movie'): {
    #             'edge_index': torch.stack([users, movies]),
    #             'edge_attr': ratings
    #         }
    #     }

    # positive/negative 샘플 생성 시 양성 샘플과 음성 샘플의 비율을 조정하고, BPR 손실을 계산할 때 이 불균형을 고려
    def prepare_train_data(self):
        user_movie_edge_index = self.data[('user', 'rates', 'movie')]['edge_index']
        user_movie_ratings = self.data[('user', 'rates', 'movie')]['edge_attr']

        self.pos_edges = user_movie_edge_index[:, user_movie_ratings >= 3]
        self.neg_edges = user_movie_edge_index[:, user_movie_ratings < 3]

        # print(f"pos_edges.shape: {self.pos_edges.shape}")
        # print(f"neg_edges.shape: {self.neg_edges.shape}")

        self.pos_sample_size = self.pos_edges.shape[1]
        self.neg_sample_size = int(self.pos_sample_size * self.pos_neg_ratio)

    def __len__(self):
        return self.pos_sample_size

    def __getitem__(self, idx):
        pos_edge = self.pos_edges[:, idx]
        neg_idx = torch.randint(0, self.neg_edges.shape[1], (1,))
        neg_edge = self.neg_edges[:, neg_idx].squeeze()
        return {'pos_edge': pos_edge, 'neg_edge': neg_edge}

    def collate_fn(self, batch):
        pos_edges = torch.stack([item['pos_edge'] for item in batch], dim=1)
        neg_edges = torch.stack([item['neg_edge'] for item in batch], dim=1)
        return {
            'pos_edge_index': pos_edges,
            'neg_edge_index': neg_edges
        }