Part 3

# Overview of Diversity in Sequential Group Recommendations

In sequential group recommendations, providing diverse recommendations is essential to enhance user satisfaction and engagement. Diversity ensures that users are exposed to a wide range of content, preventing redundancy and mitigating the reinforcement of biases that may arise from repeated exposure to similar items.

## Importance of Diversity:

- User Satisfaction: Diverse recommendations keep users interested by introducing new and varied content.
- Fairness: Ensures that the preferences of all group members are considered, especially in groups with diverse tastes.
- Preventing Filter Bubbles: Exposes users to a broader set of items, preventing over-specialization.
- Maintaining Engagement: Avoids user fatigue caused by repetitive recommendations over multiple iterations.

## Challenges in Incorporating Diversity

- Balancing Relevance and Diversity: Introducing diversity should not significantly compromise the relevance of recommendations to the group's preferences.
- Computational Complexity: Calculating diversity metrics, especially with large datasets, can be resource-intensive.
- Defining Diversity: Determining an appropriate measure of diversity that aligns with user expectations.

## Designing a New Method for Diverse Sequential Group Recommendations

We introduce a new method that incorporates Maximal Marginal Relevance (MMR) to balance relevance and diversity in sequential group recommendations. The method leverages movie genre information to compute diversity between items.

### Key Ideas:

- Genre-Based Diversity: Use the genres of movies to calculate diversity between recommended items.
- MMR Approach: Combine relevance and diversity scores to select items that are both relevant and diverse.
- Adjustable Diversity Parameter: Introduce a parameter 𝜆diversity to control the trade-off between relevance and diversity.

### Algorithm Steps:

#### Initialization:

- Load Movie Data: Import movies.csv to access movie titles and genres.
- Process Genres: Map each movieId to its set of genres for diversity calculations.

#### At Each Iteration 𝑗:

1. Compute Individual Predictions:
   - For each user in the group, predict ratings for all candidate movies using the collaborative filtering approach implemented earlier.
2. Calculate Relevance Scores:
   - For each candidate movie, calculate a relevance score based on the weighted predicted ratings from all group members.
3. Calculate Diversity Scores:
   - For each candidate movie, compute the diversity score relative to previously recommended movies using the Jaccard distance between genres.
4. Normalize Scores:
   - Normalize relevance and diversity scores to bring them to a comparable scale (e.g., between 0 and 1).
5. Compute Combined Scores:
   - Calculate the combined score for each candidate movie using the adjustable diversity parameter.
6. Select Top 𝑘 Movies:
   - Recommend the top 𝑘 movies with the highest combined scores.
7. Update User Satisfaction and Weights:
   - Calculate user satisfaction based on the group recommendation.
   - Update cumulative satisfaction and adjust user weights to adapt to changes in satisfaction over iterations.

## Why This Method Works Well

- Balances Relevance and Diversity: The MMR approach effectively combines relevance and diversity, ensuring that recommendations are both appealing and varied.
- Prevents Redundancy: By considering diversity scores, the method avoids recommending similar movies in successive iterations.
- Adaptive to User Preferences: User weights are updated based on satisfaction, ensuring that less satisfied users have more influence in future recommendations.
- Utilizes Content Information: Incorporating genre data provides a meaningful measure of diversity aligned with user perceptions.

## Instructions on How to Run the Code

### Dependencies

- Python 3.x
- Required Libraries: pandas, numpy, math, os
- Install the required libraries using: `pip install pandas numpy`

### Data Setup

- Dataset: MovieLens 100K dataset.
- Place the following CSV files in the `../data/` directory relative to the script:
  - ratings.csv
  - movies.csv
- Ensure that the directory structure is as follows:

```
your_script_directory/
├── part3
|   ├── readme.md <-- This File
│   ├── part3.pptx <-- Presentation Slides
│   └── part3.py <-- Python Script for Sequential Group Recommendations with Diversity
└── data/
     ├── ratings.csv
     └── movies.csv
```

### Running the Code

1. Navigate to the Script Directory: `cd your_script_directory`
2. Execute the Script: `python part3.py`
3. Adjust Parameters (Optional):
   - Open `part3.py` in a text editor.
   - Modify the following parameters as needed:
     - `group_users`: List of user IDs in the group
     - `iterations`: Number of recommendation rounds
     - `top_n`: Number of movies to recommend per round
     - `lambda_diversity`: Trade-off parameter between relevance and diversity
     - `beta`: Smoothing parameter for cumulative satisfaction
     - `gamma`: Parameter controlling influence of satisfaction change on weights

### Output

The script will display:

- Group Recommendations: Movie titles recommended in each iteration.
- User Satisfaction: Satisfaction scores for each user at each iteration.
- Weights: Updated user weights after each iteration.
- Overall Metrics:
  - Group Satisfaction: Average satisfaction across all users.
  - Group Disagreement: Difference between the most and least satisfied users.

## Limitations and Future Improvements

### Limitations

- Genre Granularity: Genres may be broad; movies within the same genre can still be quite different.
- Computational Overhead: Calculating diversity scores adds computational complexity, especially with large groups or datasets.

### Future Improvements

- Enhanced Diversity Metrics: Incorporate additional features like movie plots, keywords, or embeddings for a more refined diversity calculation.
- User Preference Modeling: Account for individual users' openness to diversity to personalize the trade-off.
- Scalability Optimizations: Implement efficient algorithms or approximate methods to handle larger datasets.

## Conclusion

By integrating diversity into sequential group recommendations, this method enhances user satisfaction and engagement. The balance between relevance and diversity ensures that recommendations are both appealing and varied, catering to the diverse preferences within a group. Adaptive mechanisms for user satisfaction and weight updates promote fairness over multiple iterations.
