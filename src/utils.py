def update_pea_graph_input(dataset, train_args):
    if dataset.name == "Movielens":
        data = dataset.data
        device = train_args['device']
        
        user2item_edge_index = data['user', 'rates', 'movie']['edge_index'].to(device)
        item2user_edge_index = user2item_edge_index.flip([0])
        item2year_edge_index = data['movie', 'in', 'year']['edge_index'].to(device)
        year2item_edge_index = item2year_edge_index.flip([0])
        item2genre_edge_index = data['movie', 'of', 'genre']['edge_index'].to(device)
        genre2item_edge_index = item2genre_edge_index.flip([0])
        tag2item_edge_index = data['tag', 'describes', 'movie']['edge_index'].to(device)
        item2tag_edge_index = tag2item_edge_index.flip([0])
        tag2user_edge_index = data['tag', 'used_by', 'user']['edge_index'].to(device)
        user2tag_edge_index = tag2user_edge_index.flip([0])

        # 메타 패스 정의
        meta_path_edge_indices = [
            # 사용자 중심 메타 패스
            [user2item_edge_index, item2user_edge_index],  # U-I-U
            [year2item_edge_index, item2user_edge_index], # Y-I-U
            [genre2item_edge_index, item2user_edge_index],  # G-I-U
            [user2tag_edge_index, tag2user_edge_index],  # U-T-U
            [item2tag_edge_index, tag2user_edge_index],  # I-T-U
            
            # 아이템(영화) 중심 메타 패스
            [item2user_edge_index, user2item_edge_index],  # I-U-I
            # [item2year_edge_index, year2item_edge_index],  # I-Y-I
            # [item2genre_edge_index, genre2item_edge_index],  # I-G-I
            [item2tag_edge_index, tag2item_edge_index],  # I-T-I
            [user2tag_edge_index, tag2item_edge_index],  # U-T-I
            [tag2user_edge_index, user2item_edge_index],  # T-U-I
        ]

        # print("Print sample meta path edge indices:")
        # for idx, edge_indices in enumerate(meta_path_edge_indices):
        #     print(f"Meta path {idx}: {edge_indices[0].shape} - {edge_indices[1].shape}")
        #     print(f"Meta path {idx}: {edge_indices}")

        if dataset.type == "25m":
            genome_tag2item_edge_index = data['genome_tag', 'describes', 'movie']['edge_index'].to(device)
            item2genome_tag_edge_index = genome_tag2item_edge_index.flip([0])
            meta_path_edge_indices.extend([
                [genome_tag2item_edge_index, item2user_edge_index],  # GT-I-U
                # [item2genome_tag_edge_index, genome_tag2item_edge_index],  # I-GT-I
            ])

        return meta_path_edge_indices
    else:
        raise NotImplementedError("Only Movielens dataset is implemented.")