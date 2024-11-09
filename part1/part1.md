# Part I: User-Based Collaborative Filtering and Group Recommendations

Due: November 9, 2024

## Task 1: Dataset Preparation (Task a)

Download and Explore the Dataset:

Action: Download the MovieLens 100K dataset from GroupLens.

Implementation:

- Use a programming language you're comfortable with (e.g., Python with Pandas).
- Read the dataset files (u.data, u.item, u.user).
- Display the first few rows to understand the data structure.
- Verify the count of ratings to ensure you have 100,000 entries.

Tip: Write a function to load and preview the data for reusability.

## 2. Implementing User-Based Collaborative Filtering (Tasks b & c)

### Pearson Correlation Function (Task b):

- Action: Implement the Pearson correlation coefficient to compute user similarities.
- Implementation:
  - Create a user-item rating matrix.
  - Write a function to calculate similarity between two users.
  - Ensure handling of cases where users have rated no common movies.
- Tip: Utilize vectorized operations for efficiency if using Python with NumPy.

### Prediction Function (Task c):

- Action: Implement the prediction function as taught in class.
- Implementation:
  - Predict a user's rating for a movie using weighted sums of ratings from similar users.
  - Consider only the top-N similar users to improve performance.
- Tip: Cache similarities to avoid redundant calculations.

## 3. Designing a New Similarity Function (Task d)

- Action: Propose and implement a novel similarity function.
- Implementation:
  - Option 1: Cosine Similarity adjusted for user rating biases.
  - Option 2: Mean Squared Difference to capture dissimilarities.
  - Option 3: Jaccard Similarity for binary preferences (liked/disliked).
- Justification:
  - Explain how your chosen function addresses limitations of Pearson correlation.
  - Discuss how it handles sparsity or captures different aspects of user preferences.
- Tip: Support your explanation with references to academic papers or articles.

## 4. Group Recommendations (Task e)

Average Method:

- Action: Aggregate individual user predictions by averaging.
- Implementation:
  - For each movie, compute the average predicted rating across group members.
- Tip: Use this method when group members have similar preferences.

Least Misery Method:

- Action: Aggregate using the minimum predicted rating.
- Implementation:
  - For each movie, take the lowest predicted rating among group members.
- Tip: This method ensures that no group member strongly dislikes a recommended movie.

## 5. Addressing Disagreements (Task f)

Defining Disagreements:

- Action: Develop a metric to quantify disagreements (e.g., standard deviation of ratings).
- Implementation:
  - Calculate the variance in predicted ratings for each movie among group members.

Proposed Method:

- Action: Adjust recommendations by penalizing high-disagreement movies.
- Implementation:
  - Introduce a weighting factor inversely proportional to disagreement.
  - Adjust aggregated scores accordingly.

Explanation:

- Highlight how this method balances group satisfaction.
- Emphasize fairness and the aim to minimize conflict within the group.

Presentation:

- Prepare 5 slides covering:
  - Introduction to the disagreement issue.
  - Your metric for measuring disagreement.
  - How your method integrates the metric.
  - Results showcasing improved group satisfaction.
  - Conclusion and potential future work.

## 6. Finalizing Part I

Testing:

- Validate your implementations with sample user groups.
- Ensure the results make logical sense.

Documentation:

- Write clear instructions on how to run your code.
- Comment your code for readability.

Submission:

- Package your code, explanations, and presentation slides.
- Submit before November 9, 2024, at 11:00 PM.
