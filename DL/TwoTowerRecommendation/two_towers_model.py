import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# 1. SETUP HYPERPARAMETERS & READ REAL DATA FROM DATA FOLDER
# Define path pointing to data directory
DATA_DIR = os.path.join(os.getcwd(), '../data')

# Read generated user/post interactions CSV
interactions_df = pd.read_csv(os.path.join(DATA_DIR, 'interactions.csv'))

# Match sizes to your generated dataset (100 users, 500 posts)
# We add +1 because your generated IDs are 1-indexed (1 to 100 and 1 to 500)
NUM_USERS = 100 + 1       
NUM_POSTS = 500 + 1       
EMBEDDING_DIM = 32     
BATCH_SIZE = 64
EPOCHS = 5

# Extract columns as numpy arrays shaped as (N, 1) to feed the embedding layers
user_ids_raw = interactions_df['user_id'].values.reshape(-1, 1).astype(np.int32)
post_ids_raw = interactions_df['post_id'].values.reshape(-1, 1).astype(np.int32)

# Build the high-performance tf.data Dataset out of live CSV rows
dataset = tf.data.Dataset.from_tensor_slices((user_ids_raw, post_ids_raw))
dataset = dataset.shuffle(buffer_size=2048).batch(BATCH_SIZE, drop_remainder=True)

# 2. DEFINE THE TWO-TOWER ARCHITECTURE
class UserTower(layers.Layer):
    def __init__(self, num_users, embedding_dim):
        super().__init__()
        self.embedding = layers.Embedding(num_users, embedding_dim, input_length=1)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(embedding_dim)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.squeeze(x, axis=1) # Remove sequence dimension
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.math.l2_normalize(x, axis=1) # Normalize embeddings

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
        return tf.math.l2_normalize(x, axis=1) # Normalize embeddings

class TwoTowerModel(Model):
    def __init__(self, num_users, num_posts, embedding_dim):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.post_tower = PostTower(num_posts, embedding_dim)
        self.temperature = 0.1 # Scales logits to sharpen contrastive learning

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
            similarity = tf.matmul(user_emb, post_emb, transpose_b=True)
            logits = similarity / self.temperature
            
            # Ground truth targets: Diagonal positions (0, 1, 2... BATCH_SIZE-1)
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

# Initialize and compile the model
model = TwoTowerModel(NUM_USERS, NUM_POSTS, EMBEDDING_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# 3. TRAINING LOOP
model.fit(dataset, epochs=EPOCHS)

# 4. OFFLINE INDEXING (Generate and cache all Post Embeddings)
# Gather all real unique post IDs spanning 1 to 500
all_post_ids = np.arange(1, NUM_POSTS).reshape(-1, 1).astype(np.int32)
cached_post_embeddings = model.post_tower(all_post_ids)

# 5. ONLINE INFERENCE (Retrieve candidates for a specific User)
target_user_id = np.array([[42]]).astype(np.int32) # Target User ID #42 from CSV

# Generate the single user embedding vector dynamically
single_user_embedding = model.user_tower(target_user_id) # Shape: (1, 32)

# Compute cosine similarity across all cached posts
scores = tf.matmul(single_user_embedding, cached_post_embeddings, transpose_b=True)
scores = tf.squeeze(scores, axis=0) # Flatten matrix to vector

# Fetch the top 5 highest scoring post indices
top_scores, top_post_indices = tf.math.top_k(scores, k=5)

# Shift indices back by 1 since our ID generation array started at ID 1 (index 0)
recommended_post_ids = [all_post_ids[idx][0] for idx in top_post_indices.numpy().tolist()]

print(f"\nRecommended Post IDs for User {target_user_id[0][0]}: {recommended_post_ids}")
print(f"Similarity Scores: {top_scores.numpy().tolist()}")
