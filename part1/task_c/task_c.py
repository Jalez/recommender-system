import pandas as pd
import numpy as np
from collections import defaultdict
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
# Load the ratings data
ratings = pd.read_csv(os.path.join(script_dir, '../../data', 'ratings.csv'))

# Step 1: Calculate the average rating for each user
user_avg_rating = ratings.groupby('userId')['rating'].mean().to_dict()

# Cache for user similarities to avoid redundant calculations
similarity_cache = {}

# Function to calculate and cache Pearson correlation between two users
def pearson_correlation(user1, user2):
    print(f"Calculating similarity between users {user1} and {user2}...")
    if (user1, user2) in similarity_cache:
        return similarity_cache[(user1, user2)]
    if (user2, user1) in similarity_cache:
        return similarity_cache[(user2, user1)]

    # Filter ratings by the two user IDs
    user1_ratings = ratings[ratings['userId'] == user1]
    user2_ratings = ratings[ratings['userId'] == user2]
    
    # Merge the two users' ratings on 'movieId' to get common ratings
    common_ratings = pd.merge(user1_ratings, user2_ratings, on='movieId', suffixes=('_user1', '_user2'))
    
    # If no common ratings, return NaN
    if len(common_ratings) == 0:
        return np.nan

    # Extract the ratings columns for calculating correlation
    user1_scores = common_ratings['rating_user1']
    user2_scores = common_ratings['rating_user2']
    
    # Calculate the Pearson correlation coefficient
    numerator = ((user1_scores - user1_scores.mean()) * (user2_scores - user2_scores.mean())).sum()
    denominator = np.sqrt(((user1_scores - user1_scores.mean())**2).sum()) * np.sqrt(((user2_scores - user2_scores.mean())**2).sum())

    if denominator == 0:
        similarity = np.nan
    else:
        similarity = numerator / denominator
    
    # Cache the similarity
    similarity_cache[(user1, user2)] = similarity
    
    print(f"Similarity between users {user1} and {user2}: {similarity}")
    
    return similarity

# Step 3: Predict a user's rating for a given movie
def predict_rating(user_a, movie_p, top_n=5):
    """
    Predict the rating for a given user and movie using collaborative filtering.
    
    Parameters:
    user_a (int): ID of the user.
    movie_p (int): ID of the movie.
    top_n (int): Number of top similar users to consider.
    
    Returns:
    float: The predicted rating for user_a on movie_p.
    """
    
    print(f"Predicting rating for user {user_a} on movie {movie_p}...")
    # Step 1: Get the average rating of user_a
    r_a_avg = user_avg_rating.get(user_a, 0)

    # Step 2: Find other users who have rated movie_p
    users_who_rated_p = ratings[ratings['movieId'] == movie_p]['userId'].unique()

    # Step 3: Calculate similarities between user_a and other users who rated movie_p
    similarities = []
    for user_b in users_who_rated_p:
        if user_b != user_a:
            sim_ab = pearson_correlation(user_a, user_b)
            if not np.isnan(sim_ab):
                similarities.append((user_b, sim_ab))

    # Step 4: Sort similar users by similarity score and select top N
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    # Step 5: Calculate the weighted sum for prediction
    numerator = 0.0
    denominator = 0.0

    for user_b, sim_ab in similarities:
        r_b_p = ratings[(ratings['userId'] == user_b) & (ratings['movieId'] == movie_p)]['rating'].values[0]
        r_b_avg = user_avg_rating.get(user_b, 0)

        numerator += sim_ab * (r_b_p - r_b_avg)
        denominator += abs(sim_ab)

    # Step 6: If the denominator is zero, return the user's average rating
    if denominator == 0:
        return r_a_avg

    # Step 7: Calculate the predicted rating
    predicted_rating = r_a_avg + (numerator / denominator)
    
    

    return predicted_rating

# Example usage
user_id = 2
movie_id = 10
predicted_rating = predict_rating(user_id, movie_id, top_n=5)
print(f"Predicted rating for user {user_id} on movie {movie_id}: {predicted_rating}")
