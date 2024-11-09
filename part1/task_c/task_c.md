Explanation of Steps

## Average Ratings Calculation (user_avg_rating):

First, we calculate and store the average rating for each user to avoid recalculating it multiple times.

## Caching Similarities (similarity_cache):

We use a dictionary (`similarity_cache`) to store user-user similarities. This prevents redundant calculations, saving computational time. If the similarity between two users is already in the cache, we use it directly.

## Pearson Correlation Calculation:

The `pearson_correlation` function is used to calculate similarity between two users. The results are cached, and common ratings are found by merging the users' rating data on `movieId`.

## Prediction Function (predict_rating):

1. Start with the average rating of the user (`r_a_avg`).
2. Identify users who have rated the movie in question (`users_who_rated_p`).
3. Calculate similarities between the target user and other users who rated the movie.
4. Select the top-N most similar users.
5. Use the weighted sum formula to predict the rating, adjusting based on deviations from average ratings of similar users.

## Handling Edge Cases:

- If there are no users who rated the movie or no similarities found, the user's average rating is used.
- If the denominator (sum of similarity scores) is zero, it means either no users were similar or they all had constant ratings. In that case, we return the user's average rating.

## Summary

This implementation calculates a predicted rating for a given user and movie based on the weighted sum of deviations from similar users. It considers the top-N similar users to ensure both accuracy and performance. The caching mechanism further improves performance by avoiding redundant similarity calculations.

Feel free to adjust `top_n` in the `predict_rating` function to fine-tune the number of users contributing to the prediction. Let me know if you need further clarification or improvements!
