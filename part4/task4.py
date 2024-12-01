import pandas as pd
import numpy as np
import os
import math
from collections import defaultdict
from sklearn.decomposition import NMF
from collections import defaultdict
np.random.seed(42)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

movies_data = {
    'movieId': [1, 2, 3, 4, 5],
    'title': [
        'Toy Story (1995)',
        'Jumanji (1995)',
        'Grumpier Old Men (1995)',
        'Waiting to Exhale (1995)',
        'Father of the Bride Part II (1995)'
    ],
    'genres': [
        'Adventure|Animation|Children|Comedy|Fantasy',
        'Adventure|Children|Fantasy',
        'Comedy|Romance',
        'Comedy|Drama|Romance',
        'Comedy'
    ]
}


# Load the movies data
movies = pd.read_csv(os.path.join(script_dir, '../data', 'movies.csv'))
# movies = pd.DataFrame(movies_data)


ratings_data = {
    'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'movieId': [1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 4, 5],
    'rating': [5.0, 4.0, 4.5, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0, 4.5, 5.0, 4.0]
}

# ratings = pd.DataFrame(ratings_data)
# Load the ratings data
ratings = pd.read_csv(os.path.join(script_dir, '../data', 'ratings.csv'))


# Merge ratings with movies to get titles (optional)
ratings = ratings.merge(movies, on='movieId')

# Create a user-item rating matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
print("user_item_matrix\n", user_item_matrix)

# Convert to NumPy array for NMF
R = user_item_matrix.values

# Normalize the ratings (optional)
R = R / 5.0  # Assuming ratings are out of 5

# Initialize the NMF model
nmf_model = NMF(
    n_components=10,
    init='random',
    random_state=42,
    max_iter=200,
    alpha_W=0.1,
    alpha_H='same',
    l1_ratio=0.0
)

# Fit the model to the user-item matrix
W = nmf_model.fit_transform(R)
H = nmf_model.components_

# Get the predicted ratings
predictions = np.dot(W, H) * 5.0  # Scale predictions back to original rating scale

def form_groups(user_ids, group_size=3):
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(user_ids)
    groups = [user_ids[i:i + group_size] for i in range(0, len(user_ids), group_size)]
    # Ensure all groups have the desired size
    groups = [group for group in groups if len(group) == group_size]
    return groups

user_ids = user_item_matrix.index.values
groups = form_groups(user_ids, group_size=3)

def get_group_recommendations(group, predictions, top_n=10):
    # Get indices of group members
    group_indices = [user_id - 1 for user_id in group]  # Adjusting for zero-based indexing
    # Get predicted ratings for group members
    group_predictions = predictions[group_indices, :]
    # Average predictions across group members
    group_avg_predictions = np.mean(group_predictions, axis=0)
    # Get top N recommended items
    recommended_item_indices = np.argsort(-group_avg_predictions)[:top_n]
    recommended_items = [user_item_matrix.columns[i] for i in recommended_item_indices]
    return recommended_items, group_avg_predictions

# Example: Get recommendations for the first group
group = groups[0]
recommended_items, group_avg_predictions = get_group_recommendations(group, predictions)
print(f"Group {group} recommended items: {recommended_items}")

def calculate_influence_scores_nmf(group, target_item_index, W, H, user_item_matrix):
    # Get group member indices
    group_indices = [user_id - 1 for user_id in group]
    # Original group latent factors
    group_W = W[group_indices, :]
    # Average user latent factors for the group
    group_W_avg = np.mean(group_W, axis=0)
    # Original target item score
    original_target_score = np.dot(group_W_avg, H[:, target_item_index])
    influence_scores = defaultdict(float)
    # For each user in the group
    for user_id in group:
        user_index = user_id - 1
        user_rated_items = np.where(user_item_matrix.iloc[user_index, :] > 0)[0]
        # For each item the user has rated
        for item_index in user_rated_items:
            # Calculate cosine similarity between item and target item latent factors
            item_factors = H[:, item_index]
            target_item_factors = H[:, target_item_index]
            similarity = np.dot(item_factors, target_item_factors) / (np.linalg.norm(item_factors) * np.linalg.norm(target_item_factors) + 1e-10)
            # Influence is proportional to user's rating times similarity
            rating = R[user_index, item_index]
            influence = rating * similarity
            influence_scores[(user_id, item_index)] = influence
    return influence_scores

def get_counterfactual_items(influence_scores, group, top_k=3):
    # Aggregate influence scores for items interacted with by multiple group members
    item_influences = defaultdict(float)
    item_user_count = defaultdict(set)
    for (user_id, item_index), influence in influence_scores.items():
        item_influences[item_index] += influence
        item_user_count[item_index].add(user_id)
    # Filter items interacted with by multiple group members
    items_common = {item for item, users in item_user_count.items() if len(users) > 1}
    # If no items are common, consider items with highest influence
    if not items_common:
        items_common = item_influences.keys()
    # Get top-k items by influence
    top_items = sorted(items_common, key=lambda x: -item_influences[x])[:top_k]
    return top_items

def generate_counterfactual_explanation(group, target_item_index, W, H, user_item_matrix):
    # Calculate influence scores
    influence_scores = calculate_influence_scores_nmf(group, target_item_index, W, H, user_item_matrix)
    # Identify items for explanation
    counterfactual_items = get_counterfactual_items(influence_scores, group)
    # Generate explanation
    item_titles = movies[movies['movieId'].isin([user_item_matrix.columns[i] for i in counterfactual_items])]['title'].tolist()
    target_item_id = user_item_matrix.columns[target_item_index]
    target_item_title = movies[movies['movieId'] == target_item_id]['title'].values[0]
    explanation = f"The recommendation of '{target_item_title}' is strongly influenced by your group's interaction with items: {', '.join(item_titles)}."
    return explanation

# Example usage
target_item_index = user_item_matrix.columns.get_loc(recommended_items[0])  # Target the first recommended item
explanation = generate_counterfactual_explanation(group, target_item_index, W, H, user_item_matrix)
print(explanation)