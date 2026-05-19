import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
import os

# --- 0. LOAD DATA FROM CSV ---
data_dir = '../data'

# Load CSV files
users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
posts_df = pd.read_csv(os.path.join(data_dir, 'posts.csv'))
interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))

# Extract user and post IDs
user_ids_csv = users_df['user_id'].values
post_ids_csv = posts_df['post_id'].values
interactions_users = interactions_df['user_id'].values
interactions_posts = interactions_df['post_id'].values

# --- 1. SETUP HYPERPARAMETERS & CONVERT DATA ---
NUM_USERS = len(user_ids_csv)          # Total unique users
NUM_POSTS = len(post_ids_csv)          # Total unique posts
EMBEDDING_DIM = 32                     # Vector space size
BATCH_SIZE = 64
EPOCHS = 5

# Convert 1-indexed IDs from CSV to 0-indexed for embedding layers
mock_users = (interactions_users - 1).reshape(-1, 1).astype(np.int32)
mock_posts = (interactions_posts - 1).reshape(-1, 1).astype(np.int32)

print(f"Loaded {NUM_USERS} users and {NUM_POSTS} posts")
print(f"Total interactions: {len(interactions_users)}")

# Create a high-performance tf.data Dataset
dataset = tf.data.Dataset.from_tensor_slices((mock_users, mock_posts))
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE, drop_remainder=True)

# --- 2. DEFINE THE TWO-TOWER ARCHITECTURE ---
class UserTower(layers.Layer):
    def __init__(self, num_users, embedding_dim):
        super().__init__()
        self.embedding = layers.Embedding(num_users, embedding_dim, input_length=1)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(embedding_dim)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.squeeze(x, axis=1)  # Remove sequence dimension
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.math.l2_normalize(x, axis=1)  # Normalize embeddings

class PostTower(layers.Layer):
    def __init__(self, num_posts, embedding_dim):
        super().__init__()
        self.embedding = layers.Embedding(num_posts, embedding_dim, input_length=1)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(embedding_dim)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.squeeze(x, axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.math.l2_normalize(x, axis=1)  # Normalize embeddings

class TwoTowerModel(Model):
    def __init__(self, num_users, num_posts, embedding_dim):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.post_tower = PostTower(num_posts, embedding_dim)
        self.temperature = 0.1  # Scales logits to sharpen contrastive learning

    def call(self, inputs):
        user_ids, post_ids = inputs
        user_emb = self.user_tower(user_ids)
        post_emb = self.post_tower(post_ids)
        return user_emb, post_emb

    def train_step(self, data):
        user_ids, post_ids = data
        
        with tf.GradientTape() as tape:
            # Forward pass through both towers
            user_emb, post_emb = self((user_ids, post_ids), training=True)
            
            # Compute dot product similarity matrix (Batch_Size x Batch_Size)
            # Row i represents User i compared against all items in the current batch
            similarity = tf.matmul(user_emb, post_emb, transpose_b=True)
            logits = similarity / self.temperature
            
            # Ground truth targets: Diagonal positions (0, 1, 2... BATCH_SIZE-1)
            # because user at index i interacted with post at index i
            labels = tf.range(tf.shape(user_ids)[0])
            
            # InfoNCE contrastive loss using Categorical Cross-Entropy
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )
            loss = tf.reduce_mean(loss)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

# --- 3. INITIALIZE AND COMPILE THE MODEL ---
model = TwoTowerModel(NUM_USERS, NUM_POSTS, EMBEDDING_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# --- 4. TRAINING LOOP ---
print("\nStarting training...")
model.fit(dataset, epochs=EPOCHS)

# --- 5. OFFLINE INDEXING (Generate and cache all Post Embeddings) ---
# Create an array of all unique post IDs (0-indexed)
all_post_ids = np.arange(NUM_POSTS).reshape(-1, 1).astype(np.int32)
cached_post_embeddings = model.post_tower(all_post_ids)
# Shape: (NUM_POSTS, 32) -> Ready to be exported to an indexer like ScaNN or FAISS

# --- 6. ONLINE INFERENCE (Retrieve candidates for a specific User) ---
target_user_id_0indexed = 41  # User #42 in CSV (0-indexed: 41)
target_user_id = np.array([[target_user_id_0indexed]]).astype(np.int32)

# Generate the single user embedding vector dynamically
single_user_embedding = model.user_tower(target_user_id)  # Shape: (1, 32)

# Compute cosine similarity across all cached posts
scores = tf.matmul(single_user_embedding, cached_post_embeddings, transpose_b=True)
scores = tf.squeeze(scores, axis=0)  # Flatten matrix to vector

# Fetch the top 5 highest scoring post indices
top_scores, top_post_indices = tf.math.top_k(scores, k=5)

# Convert 0-indexed predictions back to 1-indexed for display (matching CSV IDs)
top_post_ids_1indexed = top_post_indices.numpy() + 1

print(f"\nRecommended Post IDs for User {target_user_id_0indexed + 1}: {top_post_ids_1indexed.tolist()}")
print(f"Similarity Scores: {top_scores.numpy().tolist()}")
