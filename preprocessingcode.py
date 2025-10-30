import pandas as pd

# Load the Kaggle data (change the path/filename as needed)
df = pd.read_csv("instagram_reach.csv")

# Make sure you know which column contains hashtags
# For Instagram Reach, it's likely 'Hashtags', which may look like: '#food #delicious #yum'
# Convert to comma-separated
df['hashtags'] = df['Hashtags'].apply(
    lambda x: ', '.join([tag.strip().lower() for tag in str(x).split() if tag.startswith('#')])
)

# Create an 'id' column starting from 1
df.insert(0, 'id', range(1, len(df)+1))

# Select only 'id' and 'hashtags' columns
df = df[['id', 'hashtags']]

# (Optional) Filter out blank hashtags
df = df[df['hashtags'].str.strip() != ''].reset_index(drop=True)

# Save to CSV for your app
df.to_csv("insta_processed.csv", index=False)
print("Saved as insta_processed.csv with real Kaggle data!")

