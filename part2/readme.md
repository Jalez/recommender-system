<!-- @format -->

# Part 2

## 2. Overview of Sequential Group Recommendations

Sequential group recommendations involve making recommendations to a group over multiple iterations (or rounds), considering past interactions to ensure fairness and satisfaction among all group members over time.

### Challenges:

- **Fairness Over Time**: Ensuring each group member is fairly satisfied across multiple recommendation rounds.
- **Balancing Satisfaction and Disagreement**: Maximizing overall group satisfaction while minimizing individual dissatisfaction.

## 3. Limitations of Existing Methods

### Average Method

- **Pros**: Considers all group members equally.
- **Cons**: May ignore minority preferences, leading to persistent dissatisfaction for some users.

### Least Misery Method

- **Pros**: Ensures that no group member is completely dissatisfied.
- **Cons**: May lead to recommendations that are only minimally acceptable to all, without being highly satisfying to any.

### Sequential Hybrid Aggregation Method

- **Pros**: Dynamically adjusts the influence of average and least misery methods based on previous satisfaction levels.
- **Cons**: The calculation of the weighting parameter (α) may not fully capture the nuances of group dynamics over time.

## 4. Designing a New Method

We will build upon the existing methods and introduce a new approach called **Satisfaction Trend Weighted Aggregation**.

### Key Ideas:

- **Dynamic Weight Adjustment**: Adjust user weights based on their satisfaction trends over time.
- **Prevent Long-Term Dissatisfaction**: Users whose satisfaction is decreasing get higher weights in future iterations.
- **Balance Group and Individual Satisfaction**: Aim to maximize overall group satisfaction while ensuring fairness to individual members.

### Algorithm Steps:

#### Initialization:

- **Cumulative Satisfaction (CumSat)**: Initialize to zero for all users.
- **Weights (w)**: Initialize equally for all users:

  \[
  w(u)\_1 = \frac{1}{|G|}
  \]

#### At Each Iteration \( j \):

1. **Calculate Individual User Recommendations**:

   - Generate top-\( k \) recommendations for each user using your existing `predict_rating` function.

2. **Aggregate Group Recommendations**:

   - Use the current weights to aggregate individual recommendations into a group recommendation list.

3. **Calculate User Satisfaction**:

   - For each user \( u \), calculate their satisfaction \( \text{sat}(u, Gr_j) \) based on the group recommendation list.

4. **Update Cumulative Satisfaction**:

   - Update cumulative satisfaction using an exponential moving average:

     \[
     \text{CumSat}(u)_j = \beta \times \text{sat}(u, Gr_j) + (1 - \beta) \times \text{CumSat}(u)_{j-1}
     \]

     where \( \beta \) is a smoothing parameter (e.g., 0.5).

5. **Calculate Satisfaction Trend**:

   - Compute the change in satisfaction:

     \[
     \Delta \text{sat}(u)_j = \text{CumSat}(u)\_j - \text{CumSat}(u)_{j-1}
     \]

6. **Update Weights**:

   - Adjust weights using an exponential function to ensure they remain positive:

     \[
     w(u)_j = w(u)_{j-1} \times e^{\gamma \times \Delta \text{sat}(u)\_j}
     \]

     where \( \gamma \) controls the influence of satisfaction change on weights.

7. **Normalize Weights**:

   - Ensure that the sum of weights equals 1:

     \[
     \sum\_{u \in G} w(u)\_j = 1
     \]

### Why This Method Works Well

- **Dynamic Adaptation**: Adapts to changes in user satisfaction over time.
- **Fairness**: Increases the weight of users whose satisfaction is decreasing, preventing consistent dissatisfaction.
- **Balance**: Balances overall group satisfaction with individual user needs.

## Instructions on How to Run the Code

1. **Dependencies**:

   - Ensure you have `pandas`, `numpy`, and `math` libraries installed.
   - Install them using:

     ```bash
     pip install pandas numpy
     ```

2. **Data**:

   - Place the **MovieLens 100K** dataset in the appropriate directory.
   - Update file paths if necessary.

3. **Running the Code**:

   - Execute the script using a Python interpreter:

     ```bash
     python part2.py
     ```

   - Adjust parameters such as `group_users`, `iterations`, `beta`, and `gamma` as needed.

### Parameters:

- **group_users**: List of user IDs in the group.
- **iterations**: Number of recommendation rounds.
- **beta**: Controls the smoothing of cumulative satisfaction (\( 0 < \beta < 1 \)).
- **gamma**: Controls the influence of satisfaction change on weights.

## Explanation of Why the Method Works Well

### Dynamic Adaptation to User Satisfaction

- Adjusts weights based on satisfaction trends, ensuring less satisfied users gain more influence.

### Balancing Group and Individual Satisfaction

- Maximizes overall group satisfaction while considering individual preferences.

### Fairness Over Time

- Ensures all users have their preferences reflected over multiple iterations.

### Adaptability

- Adapts to changes in user preferences and satisfaction by updating weights and cumulative satisfaction each iteration.
