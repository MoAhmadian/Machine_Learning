import pandas as pd
import numpy as np

# Guarantee identical data generation on every run
np.random.seed(42)

NUM_USERS = 100
NUM_POSTS = 500
NUM_INTERACTIONS = 2500

# --- 1. GENERATE USERS METADATA ---
user_ids = np.arange(1, NUM_USERS + 1)
ages = np.random.randint(18, 50, size=NUM_USERS)
# Split gender 50/50 strictly between Male and Female
genders = np.random.choice(['Male', 'Female'], size=NUM_USERS, p=[0.50, 0.50])

users_df = pd.DataFrame({
    'user_id': user_ids,
    'age': ages,
    'gender': genders
})

# --- 2. GENERATE POSTS METADATA ---
post_ids = np.arange(1, NUM_POSTS + 1)
categories = ['Fashion', 'Travel', 'Food', 'Fitness', 'Tech', 'Memes', 'Art', 'Music']
post_categories = np.random.choice(categories, size=NUM_POSTS)
likes_count = np.random.randint(5, 10000, size=NUM_POSTS)

posts_df = pd.DataFrame({
    'post_id': post_ids,
    'category': post_categories,
    'historical_likes': likes_count
})

# --- 3. GENERATE INTERACTION LOGS ---
interacted_users = np.random.randint(1, NUM_USERS + 1, size=NUM_INTERACTIONS)
interacted_posts = np.random.randint(1, NUM_POSTS + 1, size=NUM_INTERACTIONS)
interaction_types = np.random.choice(['like', 'save', 'share'], size=NUM_INTERACTIONS, p=[0.70, 0.20, 0.10])

interactions_df = pd.DataFrame({
    'user_id': interacted_users,
    'post_id': interacted_posts,
    'interaction_type': interaction_types
})

# --- 4. EXPORT TO CSV ---
users_df.to_csv('users.csv', index=False)
posts_df.to_csv('posts.csv', index=False)
interactions_df.to_csv('interactions.csv', index=False)

print("Files 'users.csv', 'posts.csv', and 'interactions.csv' have been re-generated successfully!")
