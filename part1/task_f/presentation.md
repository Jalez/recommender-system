Slide 1: Introduction to the Problem of Disagreements in Group Recommendations

## Group Recommendation Challenges:

Recommending content to groups often involves conflicting individual preferences. Members can have varying tastes, leading to disagreements on what content is suitable.

## Importance of Addressing Disagreements:

Without careful handling, recommendations may lead to dissatisfaction. Ignoring disagreements can result in suboptimal choices that alienate some members.

## Goal:

Develop a fair recommendation system that considers individual differences to achieve group satisfaction.

## Slide 2: The Disagreement Metric Defined

**Disagreement Metric: Standard Deviation of Predicted Ratings**

- Measures the extent of disagreement among group members on predicted ratings for each movie.
- High Standard Deviation: Indicates varied opinions, implying significant disagreement.
- Low Standard Deviation: Indicates consensus among group members.

**Why Use Standard Deviation?**

- It effectively quantifies the dispersion of preferences.
- Helps identify items that are polarizing versus those that are widely accepted by the group.

## Slide 3: Integrating the Disagreement Metric

**Adjusted Average Method:**

- Start with the average predicted rating for each movie.
- Adjustment with Disagreement: Subtract a fraction of the disagreement (controlled by alpha) from the average rating.
- Adjusted Rating = Avg Rating - (alpha \* Disagreement)

**Purpose:**

- Reduce the likelihood of recommending movies with high disagreement.
- Ensure that recommendations reflect a balance between overall preference and minimizing dissatisfaction.

**Parameter alpha:**

- Controls the impact of disagreement.
- Allows for flexibility based on group dynamics.

## Slide 4: Results Showing Improved Group Recommendations

**Evaluation:**

- Compared recommendations using: Baseline (Average Method) vs. Adjusted Method (Considering Disagreement).

**Key Findings:**

- Average Method: Often recommended movies with high disagreement, leading to potential dissatisfaction.
- Adjusted Method: Recommended movies that had a higher level of consensus among group members. Reduced dissatisfaction by avoiding polarizing movies.

**Example:**

- For a diverse group, movies with lower standard deviation were more likely to be included, leading to greater overall satisfaction.

## Slide 5: Conclusion and Future Work

**Conclusion:**

- Incorporating the disagreement metric helps create fair and satisfactory group recommendations.
- The Adjusted Average Method balances high predicted ratings with low disagreement to ensure recommendations that cater to all group members.

**Benefits:**

- Promotes group harmony by avoiding movies likely to cause dissatisfaction.
- Flexible adjustment based on group preferences (alpha parameter).

**Future Work:**

- Dynamic Alpha Adjustment: Automate the selection of alpha based on group member feedback or interaction history.
- Hybrid Approaches: Combine disagreement with other aggregation methods, like Least Misery, to further enhance personalization.
- User Feedback Loop: Incorporate real-time feedback from group members to refine recommendations iteratively.
