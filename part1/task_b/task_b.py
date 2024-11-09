import pandas as pd
import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the ratings data
ratings_path = os.path.join(script_dir, '../../data', 'ratings.csv')
ratings = pd.read_csv(ratings_path)

def pearson_correlation(user1, user2):
    """
    Calculate the Pearson correlation coefficient between two users based on their movie ratings.
    
    Parameters:
    user1 (int): ID of the first user.
    user2 (int): ID of the second user.
    
    Returns:
    float: Pearson correlation coefficient between the two users. NaN if there are no common ratings.
    """
    # Step 1: Filter ratings by the two user IDs
    user1_ratings = ratings[ratings['userId'] == user1]
    user2_ratings = ratings[ratings['userId'] == user2]
    
    # Step 2: Merge the two users' ratings on 'movieId' to get common ratings
    common_ratings = pd.merge(user1_ratings, user2_ratings, on='movieId', suffixes=('_user1', '_user2'))

    # Step 3: Check if there are any movies rated in common
    if len(common_ratings) == 0:
        return np.nan  # No common ratings

    # Step 4: Extract the ratings columns for calculating correlation
    user1_scores = common_ratings['rating_user1']
    user2_scores = common_ratings['rating_user2']
    
    # Step 5: Calculate the Pearson correlation coefficient
    numerator = ((user1_scores - user1_scores.mean()) * (user2_scores - user2_scores.mean())).sum()
    denominator = np.sqrt(((user1_scores - user1_scores.mean())**2).sum()) * np.sqrt(((user2_scores - user2_scores.mean())**2).sum())

    if denominator == 0:
        return np.nan  # If denominator is zero, correlation is undefined (users have constant ratings)
    
    return numerator / denominator

# Function to find two users with at least one movie in common
def find_users_with_common_movies():
    # Step 1: Find users who have rated at least one common movie
    user_movie_groups = ratings.groupby('movieId')['userId'].unique()
    for users in user_movie_groups:
        if len(users) > 1:
            return users[0], users[1]  # Return a pair of users that have at least one movie in common

    return None, None

# Example usage
user_id_1, user_id_2 = find_users_with_common_movies()

if user_id_1 is None or user_id_2 is None:
    print("No users with common movie ratings found.")
else:
    similarity_score = pearson_correlation(user_id_1, user_id_2)
    print(f"Pearson correlation between user {user_id_1} and user {user_id_2}: {similarity_score}")
