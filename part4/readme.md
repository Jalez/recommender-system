<!-- @format -->

# Part 4

## Overview of Counterfactual Explanations in Group Recommendations

In group recommendation systems, providing explanations for recommendations enhances transparency and user trust. Counterfactual explanations, in particular, offer insights into how different user interactions influence the recommendations. They answer "what if" scenarios by identifying how changes in user interactions could alter the recommendations.

### Importance of Counterfactual Explanations

- **Transparency**: Helps users understand why certain items are recommended.
- **Fairness**: Ensures that explanations reflect the group's collective influence rather than singling out individuals.
- **User Trust**: Builds confidence in the recommendation system by providing clear reasoning.
- **Engagement**: Encourages users to interact more with the system when they understand the impact of their actions.

### Challenges in Generating Counterfactual Explanations

- **Computational Complexity**: Retraining models after removing items is resource-intensive, especially with large datasets.
- **Fairness**: Avoiding explanations that single out individual users in group settings.
- **Accuracy vs. Efficiency**: Balancing the need for precise explanations with practical computational constraints.
- **Group Dynamics**: Accounting for the diverse preferences and interactions within a group.

## Designing a New Method for Counterfactual Explanations in Group Recommendations

We introduce a novel method that generates counterfactual explanations for group recommendations using influence scores derived from Non-negative Matrix Factorization (NMF). The method focuses on ensuring fairness by considering items that multiple group members have interacted with.

### Key Ideas

- **Influence Approximation**: Estimate the influence of each item on the recommendation without retraining the model.
- **Latent Factors Utilization**: Use latent factors from NMF to calculate similarities and influence scores.
- **Fairness in Explanations**: Prioritize items interacted with by multiple group members to avoid singling out individuals.
- **Scalability**: Provide a practical solution that is computationally efficient for large datasets.

### Algorithm Steps

1. **Data Preprocessing**

- **Load Data**: Import `ratings.csv` and `movies.csv` to access user ratings and movie information.
- **Create User-Item Matrix**: Generate a matrix where rows represent users, columns represent movies, and values are ratings.
- **Normalize Ratings**: Scale ratings to a range of [0, 1] to facilitate NMF processing.

2. **Model Training with NMF**

- **Initialize NMF Model**: Set the number of latent factors and regularization parameters.
- **Fit the Model**: Decompose the user-item matrix into user and item latent factors (W and H matrices).
- **Generate Predictions**: Reconstruct the approximate user-item matrix using the latent factors. Scale predictions back to the original rating scale.

3. **Forming Groups and Generating Recommendations**

- **Form Groups**: Randomly select groups of users with a specified group size.
- **Generate Group Recommendations**: For each group, average the predicted ratings of group members. Recommend top items based on the highest average predicted ratings.

4. **Calculating Influence Scores**

- **Compute Original Target Score**: Calculate the group's average latent factors. Compute the score for the target item using the group's average latent factors.
- **Estimate Influence of Each Item**: For each item in the group's interaction history:
  - Calculate the cosine similarity between the item's latent factors and the target item's latent factors.
  - Multiply the similarity by the user's rating to estimate influence.
- **Aggregate Influence Scores**: Sum the influence scores for each item across all group members.

5. **Generating Counterfactual Explanations**

- **Identify Key Items**: Select items with high influence scores that have been interacted with by multiple group members.
- **Create Explanation**: Formulate a statement highlighting how the group's interactions with these items influenced the recommendation.
  - Example: "The recommendation of 'Inception (2010)' is strongly influenced by your group's interaction with 'The Matrix (1999)' and 'Interstellar (2014)'."

### Why This Method Works Well

- **Efficiency**: Avoids the need to retrain the model for each item removal, making it practical for large datasets.
- **Fairness**: Focuses on items interacted with by multiple group members, ensuring no individual is singled out.
- **Transparency**: Provides clear explanations based on measurable influence scores derived from the model's latent factors.
- **Scalability**: Utilizes efficient computations, suitable for real-time applications.

## Instructions on How to Run the Code

### Dependencies

- **Python 3.x**

**Required Libraries**:

- `pandas`
- `numpy`
- `scikit-learn` (version 0.24 or later)

**Install the required libraries**:

```sh
pip install pandas numpy scikit-learn
```

### Data Setup

**Dataset**: MovieLens Latest Small dataset.

**Files Needed**:

- `ratings.csv`
- `movies.csv`

**Directory Structure**:

```
your_script_directory/
├── part4
│   ├── readme.md <-- This File
│   ├── part4.pptx <-- Presentation Slides
│   └── part4.py <-- Python Script for Counterfactual Explanations
└── data/
   ├── ratings.csv
   └── movies.csv
```

_Note: Ensure the data directory contains the necessary CSV files._

### Running the Code

**Navigate to the Script Directory**:

```sh
cd part4
```

**Execute the Script**:

```sh
python part4.py
```

**Adjust Parameters (Optional)**:

- Open `part4.py` in a text editor.
- Modify the following parameters as needed:
  - **Group Size**: Change the `group_size` parameter in the `form_groups` function.
  - **Number of Latent Factors**: Adjust `n_components` in the NMF model initialization.
  - **Regularization Parameters**: Modify `alpha_W`, `alpha_H`, and `l1_ratio` in the NMF model.
  - **Random Seed**: Set `random_state` for reproducibility.

### Output

The script will display:

- **User-Item Matrix**: Prints the user-item rating matrix used for training.
- **Group Recommendations**: Shows the group formed and the top recommended items.
- **Counterfactual Explanation**: Provides an explanation highlighting the group's influential interactions.

**Example**:

```
The recommendation of 'Inception (2010)' is strongly influenced by your group's interaction with 'The Matrix (1999)', 'Interstellar (2014)'.
```

### Understanding the Output

- **Group Members**: The user IDs included in the group.
- **Recommended Items**: A list of movie IDs recommended to the group.
- **Explanation**: A statement that connects the recommended item to the group's past interactions.

## Limitations and Future Improvements

### Limitations

- **Approximation of Influence**: Influence scores are estimates and may not capture the exact impact of removing items.
- **Data Biases**: Underlying biases in the data may affect the fairness of recommendations and explanations.
- **Assumption of Linear Contribution**: The method assumes a linear relationship between item similarity and influence, which may oversimplify complex interactions.

### Future Improvements

- **Enhanced Influence Modeling**: Develop more sophisticated methods to estimate influence, possibly incorporating interaction effects.
- **Dynamic Group Preferences**: Account for temporal changes in group preferences and individual openness to influence.
- **Incorporate Additional Features**: Use content-based features or social connections to refine influence calculations.
- **Bias Mitigation Techniques**: Implement strategies to detect and reduce biases in recommendations and explanations.

## Conclusion

This method provides a practical and fair approach to generating counterfactual explanations in group recommendation systems. By leveraging NMF and influence scores, it efficiently identifies the items that collectively influence group recommendations. The focus on fairness ensures that explanations reflect the group's shared preferences, enhancing transparency and user trust.

### Additional Notes

- **Extensibility**: The method can be extended to other domains beyond movies, wherever user-item interaction data is available.
- **Customization**: Parameters can be tuned to adjust the sensitivity of influence scores and the granularity of explanations.
- **Integration**: The method can be integrated into existing recommendation systems to provide explanations without significant overhead.

### Contact Information

For any questions or assistance, please contact:

- **Name**: [Your Name]
- **Email**: [Your Email]
- **Institution**: [Your University or Organization]

### References

- **MovieLens Dataset**: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
- **Scikit-Learn NMF Documentation**: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

_End of Document_
