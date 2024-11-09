# Task B: Pearson Correlation Coefficient

## Loading the Ratings Data:

The ratings data is loaded from the `ratings.csv` file. Each row contains `userId`, `movieId`, `rating`, and `timestamp`.

### Function Definition:

`pearson_correlation(user1, user2)` takes in two user IDs as input parameters and calculates the Pearson correlation coefficient between them.

### Filtering Ratings for Each User:

We first filter the dataset to extract the ratings given by `user1` and `user2` using the condition `ratings['userId'] == userX`.

### Finding Common Ratings:

To find the movies rated by both users, we perform an inner merge on the `movieId` column, which results in a dataset that contains only the movies that both users have rated.
If there are no common movies, the function returns `np.nan` to indicate that a correlation cannot be calculated.

### Extracting Ratings Columns:

Extract the ratings of `user1` and `user2` from the merged dataset.

### Calculating the Pearson Correlation Coefficient:

The Pearson correlation formula is implemented directly:
Numerator: The covariance between the users' ratings, calculated as the sum of the product of deviations from the mean for both users.
Denominator: The product of the standard deviations of both users' ratings.
If the denominator is zero (meaning at least one of the users has constant ratings for all common movies), the function returns `np.nan` as the correlation is undefined.
Finally, the correlation value is returned.

### Handling Edge Cases:

No Common Movies: The function returns `np.nan` if the two users have no movies rated in common.
Constant Ratings: If either user has given the same rating to all common movies, the standard deviation will be zero, resulting in a denominator of zero. In this case, the correlation is undefined, and the function returns `np.nan`.

### `find_users_with_common_movies` Function:

This function finds two users who have rated at least one common movie. It:

- Groups the dataset by `movieId` and finds unique users for each movie.
- Checks if there are at least two users for a given movie. If found, it returns the first two users.
  This guarantees that the users selected have overlapping movies, avoiding `NaN` due to no common ratings.

### Improved User Selection:

Instead of using fixed user IDs (1 and 2), the new code automatically finds two users with common ratings. This makes the example more reliable.
