import pandas as pd
import numpy as np
import os
# `predict_rating` function from the previous task is already defined in ../task_d/task_d.py
# from task_d.task_d import predict_rating




# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sample data: Ratings data loaded from MovieLens dataset
ratings = pd.read_csv(os.path.join(script_dir, '../../data', 'ratings.csv'))

# Step 1: Calculate the average rating for each user
user_avg_rating = ratings.groupby('userId')['rating'].mean().to_dict()

# Cache for user similarities to avoid redundant calculations
similarity_cache = {}




# Function to calculate and cache Pearson correlation between two users
def pearson_correlation(user1, user2):
    # print(f"Calculating similarity between users {user1} and {user2}...")
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
    
    # print(f"Similarity between users {user1} and {user2}: {similarity}")
    
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
    
    # print(f"Predicting rating for user {user_a} on movie {movie_p}...")
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


# Function to calculate group disagreement metric (standard deviation)
def calculate_disagreement(group_users, movie_id):
    """
    Calculate the disagreement (standard deviation) of predicted ratings for a movie among a group of users.
    
    Parameters:
    group_users (list): List of user IDs in the group.
    movie_id (int): ID of the movie to consider.
    
    Returns:
    float: Standard deviation of predicted ratings for the movie among the group.
    """
    predicted_ratings = [predict_rating(user, movie_id) for user in group_users]
    return np.nanstd(predicted_ratings)  # Use nanstd to handle missing predictions

# Function to recommend movies to a group using an adjusted average method that considers disagreement
def group_recommendations_adjusted_method(group_users, movie_ids, top_n=5, alpha=0.5):
    """
    Recommend movies to a group of users using an adjusted average method that incorporates disagreement.
    
    Parameters:
    group_users (list): List of user IDs in the group.
    movie_ids (list): List of movie IDs to consider.
    top_n (int): Number of top movies to recommend.
    alpha (float): Weight factor to control the effect of disagreement (0 <= alpha <= 1).
    
    Returns:
    list: Top N recommended movies for the group.
    """
    group_movie_ratings = {}

    for movie in movie_ids:
        # Step 1: Calculate the average predicted rating for the movie
        predicted_ratings = [predict_rating(user, movie) for user in group_users]
        avg_rating = np.nanmean(predicted_ratings)  # Use nanmean to handle missing predictions

        # Step 2: Calculate the disagreement metric (standard deviation)
        disagreement = calculate_disagreement(group_users, movie)

        # Step 3: Adjust the average rating by subtracting a fraction of the disagreement
        # The alpha parameter controls how much the disagreement affects the final score
        adjusted_rating = avg_rating - (alpha * disagreement)

        group_movie_ratings[movie] = adjusted_rating

    # Step 4: Sort movies by the adjusted rating and return the top N
    sorted_movies = sorted(group_movie_ratings.items(), key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, rating in sorted_movies[:top_n]]

# Example usage
group_users = [1, 2, 3]  # Example group consisting of user IDs 1, 2, and 3
movie_ids = ratings['movieId'].unique()[:20]  # Consider the first 20 unique movies for simplicity

# Group recommendations using the adjusted method that considers disagreement
top_movies_adjusted = group_recommendations_adjusted_method(group_users, movie_ids, top_n=5, alpha=0.5)
print(f"Top 5 recommended movies for the group (Adjusted Average Method considering disagreement): {top_movies_adjusted}")
