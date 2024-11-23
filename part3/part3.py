import pandas as pd
import numpy as np
import os
import math

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the ratings data
ratings = pd.read_csv(os.path.join(script_dir, '../data', 'ratings.csv'))

# Load the movies data
movies = pd.read_csv(os.path.join(script_dir, '../data', 'movies.csv'))

# Process genres
movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
# Create a dictionary mapping movieId to set of genres
movie_genres = movies.set_index('movieId')['genres'].to_dict()

# Also create a mapping from movieId to title
movie_titles = movies.set_index('movieId')['title'].to_dict()

# Pre-calculate the average rating for each user
user_avg_rating = ratings.groupby('userId')['rating'].mean().to_dict()

# Cache for user similarities to avoid redundant calculations
similarity_cache = {}
predicted_ratings_cache = {}

def get_cached_prediction(user, movie):
    """Cache predictions for user-movie pairs"""
    key = (user, movie)
    if key not in predicted_ratings_cache:
        predicted_ratings_cache[key] = predict_rating(user, movie)
    return predicted_ratings_cache[key]

def generate_user_preference_lists(group_users, movie_ids, top_n):
    """Generate preference lists once for all methods"""
    user_preference_lists = {}
    for user in group_users:
        user_predictions = {movie: get_cached_prediction(user, movie) 
                            for movie in movie_ids}
        sorted_predictions = sorted(user_predictions.items(), 
                                    key=lambda x: x[1], reverse=True)
        user_preference_lists[user] = [movie_id for movie_id, _ 
                                       in sorted_predictions[:top_n]]
    return user_preference_lists

def pearson_correlation(user1, user2):
    """
    Calculate the Pearson correlation coefficient between two users.
    """
    if (user1, user2) in similarity_cache:
        return similarity_cache[(user1, user2)]
    if (user2, user1) in similarity_cache:
        return similarity_cache[(user2, user1)]
    # Filter ratings by the two user IDs
    user1_ratings = ratings[ratings['userId'] == user1]
    user2_ratings = ratings[ratings['userId'] == user2]
    # Merge the two users' ratings on 'movieId' to get common ratings
    common_ratings = pd.merge(user1_ratings, user2_ratings, on='movieId', suffixes=('_user1', '_user2'))
    # If no common ratings, return 0
    if len(common_ratings) == 0:
        similarity = 0.0
        similarity_cache[(user1, user2)] = similarity
        return similarity
    # Extract the ratings columns for calculating correlation
    user1_scores = common_ratings['rating_user1']
    user2_scores = common_ratings['rating_user2']
    # Calculate the Pearson correlation coefficient
    numerator = ((user1_scores - user1_scores.mean()) * (user2_scores - user2_scores.mean())).sum()
    denominator = np.sqrt(((user1_scores - user1_scores.mean())**2).sum()) * np.sqrt(((user2_scores - user2_scores.mean())**2).sum())
    if denominator == 0:
        similarity = 0.0
    else:
        similarity = numerator / denominator
    # Cache the similarity
    similarity_cache[(user1, user2)] = similarity
    return similarity

def predict_rating(user_a, movie_p, top_n=5):
    """
    Predict the rating for a given user and movie using collaborative filtering.
    """
    # Step 1: Get the average rating of user_a
    r_a_avg = user_avg_rating.get(user_a, 0)
    # Step 2: Find other users who have rated movie_p
    users_who_rated_p = ratings[ratings['movieId'] == movie_p]['userId'].unique()
    # Step 3: Calculate similarities between user_a and other users who rated movie_p
    similarities = []
    for user_b in users_who_rated_p:
        if user_b != user_a:
            sim_ab = pearson_correlation(user_a, user_b)
            if not np.isnan(sim_ab) and sim_ab > 0:
                similarities.append((user_b, sim_ab))
    # If no similar users found, return the user's average rating
    if not similarities:
        return r_a_avg
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

def initialize_group_users(group_users):
    """
    Initialize the cumulative satisfaction and weights for group users.
    """
    cum_satisfaction = {user: 0.0 for user in group_users}
    weights = {user: 1.0 / len(group_users) for user in group_users}
    return cum_satisfaction, weights

def calculate_user_satisfaction(user, group_recommendation, user_preference_list):
    """
    Calculate user satisfaction.
    """
    group_list_sat = sum(get_cached_prediction(user, movie) 
                         for movie in group_recommendation)
    user_list_sat = sum(get_cached_prediction(user, movie) 
                        for movie in user_preference_list)
    return group_list_sat / user_list_sat if user_list_sat != 0 else 0.0

def update_cumulative_satisfaction(user, current_satisfaction, previous_cum_satisfaction, beta):
    return beta * current_satisfaction + (1 - beta) * previous_cum_satisfaction

def update_weights(weights, cumulative_satisfaction, previous_cumulative_satisfaction, gamma):
    new_weights = {}
    total_weight = 0.0
    for user in weights:
        delta_sat = cumulative_satisfaction[user] - previous_cumulative_satisfaction[user]
        new_weight = weights[user] * math.exp(gamma * delta_sat)
        new_weights[user] = new_weight
        total_weight += new_weight
    # Normalize weights
    for user in new_weights:
        new_weights[user] /= total_weight
    return new_weights

def genre_jaccard_distance(movie1, movie2):
    """
    Compute the Jaccard distance between the genres of two movies.
    """
    genres1 = set(movie_genres.get(movie1, []))
    genres2 = set(movie_genres.get(movie2, []))
    if not genres1 and not genres2:
        return 0.0  # If both have no genres, consider distance zero
    intersection = genres1.intersection(genres2)
    union = genres1.union(genres2)
    jaccard_similarity = len(intersection) / len(union) if union else 0
    jaccard_distance = 1 - jaccard_similarity
    return jaccard_distance

def aggregate_group_recommendations_with_diversity(group_users, movie_ids, weights, top_n=5, previously_recommended=[], lambda_diversity=0.5):
    '''
    Aggregate individual recommendations from group users to generate group recommendations with diversity.
    '''
    relevance_scores = {}
    diversity_scores = {}
    for movie in movie_ids:
        if movie in previously_recommended:
            continue  # Skip movies already recommended
        relevance_score = 0.0
        for user in group_users:
            p_ui = predict_rating(user, movie)
            relevance_score += weights[user] * p_ui
        relevance_scores[movie] = relevance_score
        # Compute diversity_score
        if previously_recommended:
            distances = [genre_jaccard_distance(movie, prev_movie) for prev_movie in previously_recommended]
            diversity_score = np.mean(distances)
        else:
            diversity_score = 0.0  # No diversity in the first iteration
        diversity_scores[movie] = diversity_score
    # Normalize relevance_scores and diversity_scores
    # Relevance scores can be in a wide range, normalize to 0-1
    relevance_values = list(relevance_scores.values())
    min_rel = min(relevance_values)
    max_rel = max(relevance_values)
    rel_range = max_rel - min_rel if max_rel - min_rel != 0 else 1
    for movie in relevance_scores:
        relevance_scores[movie] = (relevance_scores[movie] - min_rel) / rel_range
    # Diversity scores are between 0 and 1 already, but let's confirm
    diversity_values = list(diversity_scores.values())
    min_div = min(diversity_values)
    max_div = max(diversity_values)
    div_range = max_div - min_div if max_div - min_div != 0 else 1
    for movie in diversity_scores:
        diversity_scores[movie] = (diversity_scores[movie] - min_div) / div_range
    # Now compute combined scores
    group_movie_scores = {}
    for movie in relevance_scores:
        combined_score = (1 - lambda_diversity) * relevance_scores[movie] + lambda_diversity * diversity_scores[movie]
        group_movie_scores[movie] = combined_score
    # Sort the movies by combined scores
    sorted_movies = sorted(group_movie_scores.items(), key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, score in sorted_movies[:top_n]]

def sequential_group_recommendations_with_diversity(group_users, all_movie_ids, iterations=5, top_n=5, beta=0.5, gamma=0.1, user_preference_lists=None, lambda_diversity=0.5):
    '''
    Generate group recommendations sequentially using user satisfaction and weights, incorporating diversity.
    '''
    # Initialize cumulative satisfaction and weights
    cum_satisfaction, weights = initialize_group_users(group_users)
    previous_cum_satisfaction = cum_satisfaction.copy()
    # Keep track of group recommendations
    group_recommendations = []
    previously_recommended = []
    previously_recommended_set = set()
    # For evaluation
    user_satisfaction_records = {user: [] for user in group_users}
    for j in range(1, iterations + 1):
        print(f"Iteration {j}")
        # Aggregate group recommendations with diversity
        group_recommendation = aggregate_group_recommendations_with_diversity(
            group_users,
            all_movie_ids,
            weights,
            top_n,
            previously_recommended,
            lambda_diversity
        )
        group_recommendations.append(group_recommendation)
        previously_recommended.extend(group_recommendation)
        previously_recommended_set.update(group_recommendation)
        # Calculate user satisfaction
        current_satisfaction = {}
        for user in group_users:
            user_satisfaction = calculate_user_satisfaction(user, group_recommendation, user_preference_lists[user])
            current_satisfaction[user] = user_satisfaction
            user_satisfaction_records[user].append(user_satisfaction)
        # Update cumulative satisfaction
        new_cum_satisfaction = {}
        for user in group_users:
            new_cum_satisfaction[user] = update_cumulative_satisfaction(
                user,
                current_satisfaction[user],
                previous_cum_satisfaction[user],
                beta
            )
        # Update weights
        weights = update_weights(weights, new_cum_satisfaction, previous_cum_satisfaction, gamma)
        # Update previous cumulative satisfaction
        previous_cum_satisfaction = new_cum_satisfaction.copy()
        # Print the satisfaction and weights for analysis
        print("User Satisfaction:", current_satisfaction)
        print("Weights:", weights)
        print("Cumulative Satisfaction:", new_cum_satisfaction)
        # Map movie IDs to titles for better readability
        print("Group Recommendation:", [movie_titles.get(mid, "Unknown") for mid in group_recommendation])
        print("-" * 50)
    # Calculate overall group satisfaction and disagreement
    group_satisfaction = np.mean([np.mean(sats) for sats in user_satisfaction_records.values()])
    group_disagreement = np.max([np.mean(sats) for sats in user_satisfaction_records.values()]) - \
                         np.min([np.mean(sats) for sats in user_satisfaction_records.values()])
    return group_recommendations, group_satisfaction, group_disagreement

# Existing Methods for Comparison
def group_recommendations_average_method(group_users, movie_ids, iterations=5, top_n=5):
    '''
    Generate group recommendations using the average method (average aggregation).
    '''
    previously_recommended = set()
    group_recommendations = []
    user_satisfaction_records = {user: [] for user in group_users}
    # Generate user preference lists
    user_preference_lists = {}
    for user in group_users:
        user_predictions = {movie: predict_rating(user, movie) for movie in movie_ids}
        sorted_user_predictions = sorted(user_predictions.items(), key=lambda x: x[1], reverse=True)
        user_preference_lists[user] = [movie_id for movie_id, rating in sorted_user_predictions[:top_n]]
    for j in range(1, iterations + 1):
        group_movie_scores = {}
        for movie in movie_ids:
            if movie in previously_recommended:
                continue
            predicted_ratings = [predict_rating(user, movie) for user in group_users]
            avg_rating = np.nanmean(predicted_ratings)
            group_movie_scores[movie] = avg_rating
        # Sort and select top_n
        sorted_movies = sorted(group_movie_scores.items(), key=lambda x: x[1], reverse=True)
        group_recommendation = [movie_id for movie_id, rating in sorted_movies[:top_n]]
        group_recommendations.append(group_recommendation)
        previously_recommended.update(group_recommendation)
        # Calculate user satisfaction
        for user in group_users:
            user_satisfaction = calculate_user_satisfaction(user, group_recommendation, user_preference_lists[user])
            user_satisfaction_records[user].append(user_satisfaction)
    # Calculate overall group satisfaction and disagreement
    group_satisfaction = np.mean([np.mean(sats) for sats in user_satisfaction_records.values()])
    group_disagreement = np.max([np.mean(sats) for sats in user_satisfaction_records.values()]) - \
                         np.min([np.mean(sats) for sats in user_satisfaction_records.values()])
    return group_recommendations, group_satisfaction, group_disagreement

def group_recommendations_least_misery_method(group_users, movie_ids, iterations=5, top_n=5):
    '''
    Generate group recommendations using the least misery method.
    '''
    previously_recommended = set()
    group_recommendations = []
    user_satisfaction_records = {user: [] for user in group_users}
    # Generate user preference lists
    user_preference_lists = {}
    for user in group_users:
        user_predictions = {movie: predict_rating(user, movie) for movie in movie_ids}
        sorted_user_predictions = sorted(user_predictions.items(), key=lambda x: x[1], reverse=True)
        user_preference_lists[user] = [movie_id for movie_id, rating in sorted_user_predictions[:top_n]]
    for j in range(1, iterations + 1):
        group_movie_scores = {}
        for movie in movie_ids:
            if movie in previously_recommended:
                continue
            predicted_ratings = [predict_rating(user, movie) for user in group_users]
            least_misery_rating = np.nanmin(predicted_ratings)
            group_movie_scores[movie] = least_misery_rating
        # Sort and select top_n
        sorted_movies = sorted(group_movie_scores.items(), key=lambda x: x[1], reverse=True)
        group_recommendation = [movie_id for movie_id, rating in sorted_movies[:top_n]]
        group_recommendations.append(group_recommendation)
        previously_recommended.update(group_recommendation)
        # Calculate user satisfaction
        for user in group_users:
            user_satisfaction = calculate_user_satisfaction(user, group_recommendation, user_preference_lists[user])
            user_satisfaction_records[user].append(user_satisfaction)
    # Calculate overall group satisfaction and disagreement
    group_satisfaction = np.mean([np.mean(sats) for sats in user_satisfaction_records.values()])
    group_disagreement = np.max([np.mean(sats) for sats in user_satisfaction_records.values()]) - \
                         np.min([np.mean(sats) for sats in user_satisfaction_records.values()])
    return group_recommendations, group_satisfaction, group_disagreement

# Main Execution
if __name__ == "__main__":
    # Group users is a list of user IDs in the group
    group_users = [1, 2, 3, 4, 5]
    print("Group Users:", group_users)
    all_movie_ids = ratings['movieId'].unique().tolist()
    print("Number of Movies:", len(all_movie_ids))
    iterations = 5
    top_n = 5
    lambda_diversity = 0.5  # Adjust this parameter as needed

    # Generate preference lists once
    print("Generating User Preference Lists...")
    user_preference_lists = generate_user_preference_lists(
        group_users, all_movie_ids, top_n)

    print("Running Group Recommendation Methods...")
    # New Method with Diversity
    group_recommendations, group_sat, group_dis = sequential_group_recommendations_with_diversity(
        group_users,
        all_movie_ids,
        iterations=iterations,
        top_n=top_n,
        beta=0.5,
        gamma=0.1,
        user_preference_lists=user_preference_lists,
        lambda_diversity=lambda_diversity
    )

    print("New Method with Diversity - Sequential Group Recommendations")
    print(f"Overall Group Satisfaction: {group_sat}")
    print(f"Group Disagreement: {group_dis}")
    print("=" * 50)

    # Run the Average Method
    avg_recommendations, avg_sat, avg_dis = group_recommendations_average_method(
        group_users,
        all_movie_ids,
        iterations=iterations,
        top_n=top_n
    )

    print("Average Method - Group Recommendations")
    print(f"Overall Group Satisfaction: {avg_sat}")
    print(f"Group Disagreement: {avg_dis}")
    print("=" * 50)

    # Run the Least Misery Method
    lm_recommendations, lm_sat, lm_dis = group_recommendations_least_misery_method(
        group_users,
        all_movie_ids,
        iterations=iterations,
        top_n=top_n
    )

    print("Least Misery Method - Group Recommendations")
    print(f"Overall Group Satisfaction: {lm_sat}")
    print(f"Group Disagreement: {lm_dis}")
    print("=" * 50)

    # Compare the methods
    print("Comparison of Methods:")
    print(f"{'Method':<30}{'Group Satisfaction':<20}{'Group Disagreement':<20}")
    print(f"{'New Method with Diversity':<30}{group_sat:<20.4f}{group_dis:<20.4f}")
    print(f"{'Average Method':<30}{avg_sat:<20.4f}{avg_dis:<20.4f}")
    print(f"{'Least Misery':<30}{lm_sat:<20.4f}{lm_dis:<20.4f}")
