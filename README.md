# Enhanced-MovieLens-Recommender

This project extends and refines the PEAGNN (Path Extrapolation and Aggregation Graph Neural Networks) approach for movie recommendations using the MovieLens dataset. Building upon the work from [PEAGNN](https://github.com/blindsubmission1/PEAGNN), primary focus is on redefining the criteria for positive and negative samples to improve recommendation accuracy.

## Key Contributions

1. **Redefined Positive/Negative Sample Criteria**: 
   - Implemented a new, more nuanced approach to classifying user interactions:
     - Positive samples: Ratings >= 4
     - Negative samples: Ratings <= 2
   - This redefinition aims to capture user preferences more accurately, considering ratings of 4 and above as indicative of positive user experience.

2. **Adaptive Sampling Strategy**:
   - Developed a method to handle the natural imbalance between positive and negative samples resulting from the new criteria.
   - Introduced a configurable positive-to-negative ratio for flexible dataset creation, allowing for experimentation with different sampling ratios.

3. **Modified Model Architecture**:
   - Adapted the PEAGAT (Path Extrapolation and Aggregation Graph Attention Network) model to work effectively with the newly defined sample criteria.
   - Updated the forward pass and prediction methods to handle the redefined positive and negative interactions.

4. **Refined Evaluation Method**:
   - Adjusted the test method to align with the new sample definition, ensuring that the evaluation reflects the updated criteria for positive interactions.
   - Implemented Hit Ratio at 10 (HR@10) as the primary evaluation metric, based on the redefined positive sample criterion.

## Implementation Details

- **Data Preprocessing**: The `MovieLens` class now includes methods for preparing training data based on the new positive/negative sample criteria.
- **Model**: The `PEAGATRecsysModel` class has been updated to work with the redefined interaction types.
- **Training and Evaluation**: The `Solver` class now incorporates the new sample definitions in both training and evaluation processes.

## Running the Code

```bash
python main.py --dataset Movielens --type 25m --hidden_dim 64 --num_layers 2 --num_heads 1 --dropout 0 --channel_aggr att --device cuda --gpu_idx 0 --runs 5 --epochs 30 --batch_size 1024 --lr 0.001 --weight_decay 0.001
```

## Future Work

- Investigate the impact of different rating thresholds for positive/negative sample classification.
- Explore how the redefined criteria affect recommendation diversity and novelty.
- Conduct comparative studies with other sample definition approaches in recommender systems.

## Project Basis

This work is based on the PEAGNN project.