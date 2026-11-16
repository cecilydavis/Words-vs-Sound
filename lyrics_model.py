# ==================================================
# STEP 34: SET UP LYRICS-ONLY MODEL
# ==================================================

import pandas as pd
import numpy as np
import torch

# Keep PyTorch on one CPU thread to avoid previous crashes
torch.set_num_threads(1)

print("Starting lyrics-only model...")

# Load dataset
df = pd.read_csv("data/spotify_dataset.csv")

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Load corrected dataset splits
train_indices = np.load("models/train_indices_grouped.npy")
val_indices = np.load("models/val_indices_grouped.npy")
test_indices = np.load("models/test_indices_grouped.npy")

# Select lyrics for each dataset
train_lyrics = df.loc[train_indices, "text"]
val_lyrics = df.loc[val_indices, "text"]
test_lyrics = df.loc[test_indices, "text"]

# Verify the results
print("\nLyrics datasets loaded successfully!")

print("Training lyrics:", len(train_lyrics))
print("Validation lyrics:", len(val_lyrics))
print("Testing lyrics:", len(test_lyrics))

print("\nExample training lyrics:")
print(train_lyrics.iloc[0][:200])

# ==================================================
# STEP 35: CONVERT LYRICS INTO NUMERICAL FEATURES
# ==================================================

from sklearn.feature_extraction.text import TfidfVectorizer

print("\nConverting lyrics into numerical features...")

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    min_df=5,
    dtype=np.float32
)

# Learn vocabulary using ONLY training lyrics
X_train_lyrics = tfidf.fit_transform(
    train_lyrics.fillna("")
)

# Convert validation lyrics using the same vocabulary
X_val_lyrics = tfidf.transform(
    val_lyrics.fillna("")
)

# Convert testing lyrics using the same vocabulary
X_test_lyrics = tfidf.transform(
    test_lyrics.fillna("")
)

# Check the results
print("\nTF-IDF conversion completed!")

print("Training shape:", X_train_lyrics.shape)
print("Validation shape:", X_val_lyrics.shape)
print("Testing shape:", X_test_lyrics.shape)

print("\nNumber of vocabulary features:", len(tfidf.vocabulary_))

# Show an example of learned vocabulary
print("\nExample vocabulary words:")
print(tfidf.get_feature_names_out()[:20])

# ==================================================
# STEP 36: INSPECT LYRICS FEATURE MATRICES
# ==================================================

print("\nInspecting lyrics feature matrices...")

# Check matrix types
print("\nTraining matrix type:", type(X_train_lyrics))
print("Validation matrix type:", type(X_val_lyrics))
print("Testing matrix type:", type(X_test_lyrics))

# Count nonzero values
print("\nNonzero training values:", X_train_lyrics.nnz)
print("Nonzero validation values:", X_val_lyrics.nnz)
print("Nonzero testing values:", X_test_lyrics.nnz)

# Calculate percentage of values that are nonzero
total_values = X_train_lyrics.shape[0] * X_train_lyrics.shape[1]

density = (X_train_lyrics.nnz / total_values) * 100

print("\nTraining matrix density:", round(density, 2), "%")

# Calculate memory usage
sparse_memory = (
    X_train_lyrics.data.nbytes
    + X_train_lyrics.indices.nbytes
    + X_train_lyrics.indptr.nbytes
)

dense_memory = total_values * 4

print("\nSparse training memory:",
      round(sparse_memory / (1024 ** 2), 2), "MB")

print("Estimated dense training memory:",
      round(dense_memory / (1024 ** 3), 2), "GB")

print("\nLyrics matrix inspection completed!")

# ==================================================
# STEP 37: CREATE LYRICS TRAINING BATCHES
# ==================================================

from torch.utils.data import Dataset, DataLoader

print("\nCreating lyrics training batches...")

# Our five activity labels
activities = [
    "Good for Party",
    "Good for Work/Study",
    "Good for Relaxation/Meditation",
    "Good for Exercise",
    "Good for Driving"
]

# Get the training labels
y_train_lyrics = df.loc[
    train_indices,
    activities
].to_numpy(dtype=np.float32)


# Custom dataset for sparse lyrics features
class LyricsDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):

        # Get one song's lyrics features
        song_features = self.features[index].toarray().ravel()

        # Get that song's activity labels
        song_labels = self.labels[index]

        # Convert both into PyTorch tensors
        return (
            torch.from_numpy(song_features),
            torch.from_numpy(song_labels)
        )


# Create training dataset
lyrics_train_dataset = LyricsDataset(
    X_train_lyrics,
    y_train_lyrics
)

# Create training DataLoader
lyrics_train_loader = DataLoader(
    lyrics_train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0
)

# Check our results
print("\nTotal training songs:", len(lyrics_train_dataset))
print("Number of training batches:", len(lyrics_train_loader))

# Get one batch to test
first_batch = next(iter(lyrics_train_loader))

batch_lyrics, batch_labels = first_batch

print("\nFirst batch lyrics shape:", batch_lyrics.shape)
print("First batch labels shape:", batch_labels.shape)

print("\nLyrics training batches created successfully!")

# ==================================================
# STEP 38: BUILD LYRICS-ONLY NEURAL NETWORK
# ==================================================

import torch.nn as nn

print("\nBuilding lyrics-only neural network...")


# Define the neural network
class LyricsModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Input layer: 5,000 lyrics features
            nn.Linear(5000, 128),
            nn.ReLU(),

            # Hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer: 5 activity predictions
            nn.Linear(64, 5)

        )

    def forward(self, x):
        return self.network(x)


# Set random seed for reproducibility
torch.manual_seed(42)

# Create the model
lyrics_model = LyricsModel()

# Display the model architecture
print("\nLyrics Model Architecture:")
print(lyrics_model)

# Test the model using our first batch
with torch.no_grad():
    test_predictions = lyrics_model(batch_lyrics)

print("\nInput shape:", batch_lyrics.shape)
print("Output shape:", test_predictions.shape)

print("\nLyrics-only neural network created successfully!")

# ==================================================
# STEP 39: SET UP LYRICS MODEL TRAINING
# ==================================================

print("\nSetting up lyrics model training...")

# Count positive examples for each activity
positive_counts = y_train_lyrics.sum(axis=0)

# Count negative examples for each activity
negative_counts = len(y_train_lyrics) - positive_counts

# Calculate class weights
lyrics_class_weights = negative_counts / np.maximum(
    positive_counts,
    1
)

# Convert class weights into PyTorch tensor
lyrics_weights_tensor = torch.tensor(
    lyrics_class_weights,
    dtype=torch.float32
)

# Create weighted loss function
lyrics_loss_function = nn.BCEWithLogitsLoss(
    pos_weight=lyrics_weights_tensor
)

# Create Adam optimizer
lyrics_optimizer = torch.optim.Adam(
    lyrics_model.parameters(),
    lr=0.001
)

# Display the results
print("\nClass weights:")

for activity, weight in zip(activities, lyrics_class_weights):
    print(activity, ":", round(float(weight), 2))

# Test loss calculation using our first batch
with torch.no_grad():

    test_outputs = lyrics_model(batch_lyrics)

    test_loss = lyrics_loss_function(
        test_outputs,
        batch_labels
    )

print("\nTest loss:", round(test_loss.item(), 4))

print("\nLyrics model training setup completed!")

# ==================================================
# STEP 40: TRAIN LYRICS-ONLY NEURAL NETWORK
# ==================================================

print("\nTraining lyrics-only neural network...")

# Train for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):

    # Put model into training mode
    lyrics_model.train()

    # Track total loss
    total_loss = 0

    # Go through each training batch
    for batch_lyrics, batch_labels in lyrics_train_loader:

        # Clear previous gradients
        lyrics_optimizer.zero_grad()

        # Make predictions
        predictions = lyrics_model(batch_lyrics)

        # Calculate loss
        loss = lyrics_loss_function(
            predictions,
            batch_labels
        )

        # Calculate gradients
        loss.backward()

        # Update model weights
        lyrics_optimizer.step()

        # Track loss
        total_loss += loss.item()

    # Calculate average loss for this epoch
    average_loss = total_loss / len(lyrics_train_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"| Average Training Loss: {average_loss:.4f}",
        flush=True
    )

print("\nLyrics-only model training completed!")

# ==================================================
# STEP 41: EVALUATE LYRICS-ONLY NEURAL NETWORK
# ==================================================

from sklearn.metrics import f1_score, classification_report

print("\nEvaluating lyrics-only neural network...")

# Get validation labels
y_val_lyrics = df.loc[
    val_indices,
    activities
].to_numpy(dtype=np.float32)

# Create validation dataset
lyrics_val_dataset = LyricsDataset(
    X_val_lyrics,
    y_val_lyrics
)

# Create validation DataLoader
lyrics_val_loader = DataLoader(
    lyrics_val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

# Put model in evaluation mode
lyrics_model.eval()

# Store predictions from each batch
all_probabilities = []

# Evaluate without updating model weights
with torch.no_grad():

    for batch_lyrics, batch_labels in lyrics_val_loader:

        # Get raw predictions
        outputs = lyrics_model(batch_lyrics)

        # Convert outputs to probabilities
        probabilities = torch.sigmoid(outputs)

        # Store probabilities
        all_probabilities.append(probabilities.numpy())

# Combine predictions from all batches
lyrics_val_probabilities = np.concatenate(
    all_probabilities,
    axis=0
)

# Apply an initial threshold of 0.5
lyrics_val_predictions = (
    lyrics_val_probabilities >= 0.5
).astype(int)

# Calculate Micro F1
lyrics_micro_f1 = f1_score(
    y_val_lyrics,
    lyrics_val_predictions,
    average="micro",
    zero_division=0
)

# Calculate Macro F1
lyrics_macro_f1 = f1_score(
    y_val_lyrics,
    lyrics_val_predictions,
    average="macro",
    zero_division=0
)

# Display results
print("\nLyrics Micro F1:", round(lyrics_micro_f1, 4))
print("Lyrics Macro F1:", round(lyrics_macro_f1, 4))

print("\nPerformance by activity:")

print(
    classification_report(
        y_val_lyrics,
        lyrics_val_predictions,
        target_names=activities,
        zero_division=0
    )
)

print("\nLyrics-only evaluation completed!")

# ==================================================
# STEP 42: OPTIMIZE LYRICS MODEL THRESHOLDS
# ==================================================

print("\nOptimizing lyrics model thresholds...")

lyrics_best_thresholds = []

# Test thresholds from 0.05 to 0.95
threshold_options = np.arange(0.05, 0.96, 0.05)

for i, activity in enumerate(activities):

    best_f1 = -1
    best_threshold = 0.5

    for threshold in threshold_options:

        # Predict activity using this threshold
        activity_predictions = (
            lyrics_val_probabilities[:, i] >= threshold
        ).astype(int)

        # Calculate F1
        score = f1_score(
            y_val_lyrics[:, i],
            activity_predictions,
            zero_division=0
        )

        # Keep the best threshold
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    lyrics_best_thresholds.append(best_threshold)

    print(
        activity,
        "| Best threshold:", round(best_threshold, 2),
        "| F1:", round(best_f1, 4)
    )

# Apply optimized thresholds
lyrics_optimized_predictions = np.zeros_like(
    y_val_lyrics,
    dtype=int
)

for i, threshold in enumerate(lyrics_best_thresholds):

    lyrics_optimized_predictions[:, i] = (
        lyrics_val_probabilities[:, i] >= threshold
    ).astype(int)

# Calculate overall scores
lyrics_optimized_micro = f1_score(
    y_val_lyrics,
    lyrics_optimized_predictions,
    average="micro",
    zero_division=0
)

lyrics_optimized_macro = f1_score(
    y_val_lyrics,
    lyrics_optimized_predictions,
    average="macro",
    zero_division=0
)

print("\nOptimized Lyrics Micro F1:", round(lyrics_optimized_micro, 4))
print("Optimized Lyrics Macro F1:", round(lyrics_optimized_macro, 4))

print("\nLyrics threshold optimization completed!")

# ==================================================
# STEP 43: SAVE CURRENT LYRICS MODEL
# ==================================================

import joblib

print("\nSaving current lyrics-only model...")

# Save trained neural network
torch.save(
    lyrics_model.state_dict(),
    "models/lyrics_model_5epochs.pth"
)

# Save TF-IDF vectorizer
joblib.dump(
    tfidf,
    "models/lyrics_tfidf.pkl"
)

# Save optimized thresholds
np.save(
    "models/lyrics_thresholds_5epochs.npy",
    np.array(lyrics_best_thresholds)
)

# Save model information
joblib.dump(
    {
        "epochs": 5,
        "input_features": 5000,
        "activities": activities,
        "micro_f1": lyrics_optimized_micro,
        "macro_f1": lyrics_optimized_macro
    },
    "models/lyrics_metadata_5epochs.pkl"
)

print("\nLyrics-only model saved successfully!")

print("Micro F1:", round(lyrics_optimized_micro, 4))
print("Macro F1:", round(lyrics_optimized_macro, 4))

# ==================================================
# STEP 44: CONTINUE LYRICS MODEL TRAINING
# ==================================================

print("\nContinuing lyrics model training...")

# Train for 5 additional epochs
additional_epochs = 5

for epoch in range(additional_epochs):

    # Put model into training mode
    lyrics_model.train()

    total_loss = 0

    for batch_lyrics, batch_labels in lyrics_train_loader:

        # Clear previous gradients
        lyrics_optimizer.zero_grad()

        # Make predictions
        predictions = lyrics_model(batch_lyrics)

        # Calculate loss
        loss = lyrics_loss_function(
            predictions,
            batch_labels
        )

        # Calculate gradients
        loss.backward()

        # Update model weights
        lyrics_optimizer.step()

        # Track loss
        total_loss += loss.item()

    average_loss = total_loss / len(lyrics_train_loader)

    print(
        f"Epoch {epoch + 6}/10 "
        f"| Average Training Loss: {average_loss:.4f}",
        flush=True
    )

print("\nAdditional lyrics training completed!")

# ==================================================
# STEP 45: EVALUATE 10-EPOCH LYRICS MODEL
# ==================================================

print("\nEvaluating 10-epoch lyrics model...")

# Put model into evaluation mode
lyrics_model.eval()

# Store validation probabilities
all_probabilities_10 = []

# Generate predictions in batches
with torch.no_grad():

    for batch_lyrics, batch_labels in lyrics_val_loader:

        outputs = lyrics_model(batch_lyrics)

        probabilities = torch.sigmoid(outputs)

        all_probabilities_10.append(
            probabilities.numpy()
        )

# Combine all predictions
lyrics_val_probabilities_10 = np.concatenate(
    all_probabilities_10,
    axis=0
)

# Find the best threshold for each activity
lyrics_best_thresholds_10 = []

threshold_options = np.arange(0.05, 0.96, 0.05)

print("\nOptimized thresholds after 10 epochs:")

for i, activity in enumerate(activities):

    best_f1 = -1
    best_threshold = 0.5

    for threshold in threshold_options:

        predictions = (
            lyrics_val_probabilities_10[:, i] >= threshold
        ).astype(int)

        score = f1_score(
            y_val_lyrics[:, i],
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    lyrics_best_thresholds_10.append(best_threshold)

    print(
        activity,
        "| Threshold:", round(best_threshold, 2),
        "| F1:", round(best_f1, 4)
    )

# Apply optimized thresholds
lyrics_predictions_10 = np.zeros_like(
    y_val_lyrics,
    dtype=int
)

for i, threshold in enumerate(lyrics_best_thresholds_10):

    lyrics_predictions_10[:, i] = (
        lyrics_val_probabilities_10[:, i] >= threshold
    ).astype(int)

# Calculate overall scores
lyrics_micro_10 = f1_score(
    y_val_lyrics,
    lyrics_predictions_10,
    average="micro",
    zero_division=0
)

lyrics_macro_10 = f1_score(
    y_val_lyrics,
    lyrics_predictions_10,
    average="macro",
    zero_division=0
)

print("\n10-Epoch Micro F1:", round(lyrics_micro_10, 4))
print("10-Epoch Macro F1:", round(lyrics_macro_10, 4))

# Compare against saved 5-epoch results
print("\nCOMPARISON:")

print("5-Epoch Micro F1:", round(lyrics_optimized_micro, 4))
print("10-Epoch Micro F1:", round(lyrics_micro_10, 4))

print("5-Epoch Macro F1:", round(lyrics_optimized_macro, 4))
print("10-Epoch Macro F1:", round(lyrics_macro_10, 4))

print("\n10-epoch evaluation completed!")