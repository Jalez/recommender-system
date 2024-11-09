Part II: Sequential Group Recommendations
Due: November 16, 2024

## 1. Understanding Sequential Recommendations

**Action:** Review class materials on sequential patterns in user behavior.

**Implementation:**

- Identify how users' recent activities influence their current preferences.
- Consider time-aware recommendation models.

## 2. Designing the Method (7 points)

**Option 1: Incorporate Time Decay Factors**

- **Action:** Assign higher weights to recent user interactions.
- **Justification:** Reflects the evolving tastes of users over time.

**Option 2: Sequential Pattern Mining**

- **Action:** Use algorithms like SPADE or PrefixSpan to discover frequent sequences.
- **Justification:** Captures common viewing patterns among users.

**Option 3: Markov Chains**

- **Action:** Model the probability of a user watching a movie based on previous movies.
- **Justification:** Accounts for immediate sequential dependencies.

## 3. Implementing the Method (7 points)

**Action:** Choose one of the above options and implement it.

**Implementation:**

- Modify your collaborative filtering model to incorporate sequential data.
- Ensure compatibility with group recommendation aggregation methods.

## 4. Explanation and Clarifications (6 points)

**Action:** Write a detailed explanation of your method.

**Content:**

- Theoretical background.
- How the method addresses the challenges in sequential group recommendations.
- Any assumptions or limitations.

## 5. Presentation (5 points)

**Slides Content:**

- Overview of sequential recommendation challenges.
- Your proposed method.
- Implementation highlights.
- Results and observations.
- Conclusion.

## 6. Finalizing Part II

**Testing:**

- Use sequences of user interactions from the dataset.
- Demonstrate improvements over non-sequential methods.

**Documentation:**

- Update instructions to include any new dependencies or steps.

**Submission:**

- Submit all materials before November 16, 2024, at 11:00 PM.
