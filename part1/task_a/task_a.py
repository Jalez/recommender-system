import pandas as pd
import os
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file paths
ratings_path = os.path.join(script_dir, '../../data', 'ratings.csv')
movies_path = os.path.join(script_dir, '../../data', 'movies.csv')
tags_path = os.path.join(script_dir, '../../data', 'tags.csv')
links_path = os.path.join(script_dir, '../../data', 'links.csv')

# Read the CSV files
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)
tags = pd.read_csv(tags_path)
links = pd.read_csv(links_path)

# Step 2: Display the first few rows of each dataset to understand their structure
print("Ratings dataset:")
print(ratings.head())

print("\nMovies dataset:")
print(movies.head())

print("\nTags dataset:")
print(tags.head())

print("\nLinks dataset:")
print(links.head())

# Step 3: Count the number of ratings in the 'ratings' file
num_ratings = len(ratings)
print(f"\nNumber of ratings in the ratings dataset: {num_ratings}")

# Verify it's over 100,000
assert num_ratings > 100000, "Number of ratings is less than 100,000"
