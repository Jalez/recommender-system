import pandas as pd
import numpy as np
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Load the ratings data
ratings = pd.read_csv(os.path.join(script_dir, '../../data', 'ratings.csv'))

# Pre-calculate the average rating for each user to avoid recalculating multiple times
user_avg_rating = ratings.groupby('userId')['rating'].mean().to_dict()

def cosine_similarity_adjusted(user1, user2):
    """
    Calculate the adjusted cosine similarity between two users based on their movie ratings.
    
    Parameters:
    user1 (int): ID of the first user.
    user2 (int): ID of the second user.
    
    Returns:
    float: Adjusted cosine similarity between the two users. NaN if there are no common ratings.
    """
    # Step 1: Filter ratings by the two user IDs
    user1_ratings = ratings[ratings['userId'] == user1]
    user2_ratings = ratings[ratings['userId'] == user2]
    
    # Step 2: Merge the two users' ratings on 'movieId' to get common ratings
    common_ratings = pd.merge(user1_ratings, user2_ratings, on='movieId', suffixes=('_user1', '_user2'))

    # Step 3: Check if there are any movies rated in common
    if len(common_ratings) == 0:
        return np.nan  # No common ratings

    # Step 4: Adjust ratings by subtracting the user's average rating
    user1_adjusted = common_ratings['rating_user1'] - user_avg_rating[user1]
    user2_adjusted = common_ratings['rating_user2'] - user_avg_rating[user2]

    # Step 5: Calculate the cosine similarity
    numerator = np.dot(user1_adjusted, user2_adjusted)
    denominator = np.sqrt(np.dot(user1_adjusted, user1_adjusted)) * np.sqrt(np.dot(user2_adjusted, user2_adjusted))

    if denominator == 0:
        return np.nan  # If denominator is zero, similarity is undefined (users have constant ratings)

    return numerator / denominator

# Example usage
user_id_1 = 1
user_id_2 = 2
similarity_score = cosine_similarity_adjusted(user_id_1, user_id_2)
print(f"Adjusted cosine similarity between user {user_id_1} and user {user_id_2}: {similarity_score}")
