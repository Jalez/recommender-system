# Task A: Data Loading and Exploration

Loading the CSV Files:

I use `pd.read_csv()` to load each dataset (`ratings.csv`, `movies.csv`, `tags.csv`, and `links.csv`) into separate pandas DataFrames.
These files are located in the `/mnt/data/` directory.

Displaying the First Few Rows:

I use `head()` on each DataFrame to print the first few rows, allowing us to understand the structure and columns of the datasets (`userId`, `movieId`, `rating`, etc. for the ratings file).

Counting the Ratings:

I use `len(ratings)` to calculate the total number of rows in the ratings DataFrame to confirm if there are 100,000 entries. The result is printed to verify the number of ratings.

Expected Output:
The ratings dataset should contain columns like `userId`, `movieId`, `rating`, `timestamp`.

The movies dataset contains information like `movieId`, `title`, `genres`.

The tags dataset has `userId`, `movieId`, `tag`, `timestamp`.

The links dataset provides `movieId`, `imdbId`, `tmdbId`.

You will also see the number of ratings, which should match the expected count (100,000).

Make sure to adjust file paths if you're running this code in a different environment. Let me know if you need further explanations or modifications!
