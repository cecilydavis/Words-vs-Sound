# ==================================================
# STEP 46: SET UP COMBINED WORDS + SOUND MODEL
# ==================================================

import pandas as pd
import numpy as np
import torch

torch.set_num_threads(1)

print("Starting combined Words + Sound model...")

# Load dataset
df = pd.read_csv("data/spotify_dataset.csv")

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Load corrected dataset splits
train_indices = np.load("models/train_indices_grouped.npy")
val_indices = np.load("models/val_indices_grouped.npy")
test_indices = np.load("models/test_indices_grouped.npy")

# Define sound features
sound_features = [
    "Tempo",
    "Loudness (db)",
    "Energy",
    "Danceability",
    "Positiveness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness"
]

# Define activity labels
activities = [
    "Good for Party",
    "Good for Work/Study",
    "Good for Relaxation/Meditation",
    "Good for Exercise",
    "Good for Driving"
]

# Select lyrics
train_lyrics = df.loc[train_indices, "text"]
val_lyrics = df.loc[val_indices, "text"]
test_lyrics = df.loc[test_indices, "text"]

# Check dataset sizes
print("\nCombined model datasets loaded!")

print("Training songs:", len(train_indices))
print("Validation songs:", len(val_indices))
print("Testing songs:", len(test_indices))

print("\nSound features:", len(sound_features))
print("Lyrics features: 5000")
print("Total planned features:", len(sound_features) + 5000)

print("\nCombined model setup completed!")

# ==================================================
# STEP 47: PREPARE SOUND FEATURES
# ==================================================

import joblib

print("\nPreparing sound features for combined model...")

# Clean the Loudness column
df["Loudness (db)"] = (
    df["Loudness (db)"]
    .astype(str)
    .str.replace("db", "", regex=False)
    .str.strip()
)

# Convert sound features to numerical values
for feature in sound_features:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

# Load the saved sound scaler
sound_scaler = joblib.load(
    "models/sound_scaler_grouped.pkl"
)

# Select sound features for each dataset
X_train_sound = df.loc[
    train_indices, sound_features
].to_numpy(dtype=np.float32)

X_val_sound = df.loc[
    val_indices, sound_features
].to_numpy(dtype=np.float32)

X_test_sound = df.loc[
    test_indices, sound_features
].to_numpy(dtype=np.float32)

# Check for missing values
print("\nMissing sound values:")
print("Training:", np.isnan(X_train_sound).sum())
print("Validation:", np.isnan(X_val_sound).sum())
print("Testing:", np.isnan(X_test_sound).sum())

# Apply the saved scaler
X_train_sound = sound_scaler.transform(X_train_sound)
X_val_sound = sound_scaler.transform(X_val_sound)
X_test_sound = sound_scaler.transform(X_test_sound)

# Display results
print("\nScaled sound feature shapes:")
print("Training:", X_train_sound.shape)
print("Validation:", X_val_sound.shape)
print("Testing:", X_test_sound.shape)

print("\nSound feature preparation completed!")

# ==================================================
# STEP 48: PREPARE LYRICS FEATURES
# ==================================================

print("\nPreparing lyrics features for combined model...")

# Load the saved TF-IDF vectorizer
lyrics_tfidf = joblib.load(
    "models/lyrics_tfidf.pkl"
)

# Convert lyrics into numerical features
X_train_lyrics = lyrics_tfidf.transform(
    train_lyrics.fillna("")
)

X_val_lyrics = lyrics_tfidf.transform(
    val_lyrics.fillna("")
)

X_test_lyrics = lyrics_tfidf.transform(
    test_lyrics.fillna("")
)

# Display dataset shapes
print("\nLyrics feature shapes:")
print("Training:", X_train_lyrics.shape)
print("Validation:", X_val_lyrics.shape)
print("Testing:", X_test_lyrics.shape)

# Check that matrices are sparse
print("\nLyrics matrix types:")
print("Training:", type(X_train_lyrics))
print("Validation:", type(X_val_lyrics))
print("Testing:", type(X_test_lyrics))

print("\nLyrics feature preparation completed!")

# ==================================================
# STEP 49: COMBINE SOUND + LYRICS FEATURES
# ==================================================

from scipy.sparse import csr_matrix, hstack

print("\nCombining sound and lyrics features...")

# Convert sound features into sparse matrices
X_train_sound_sparse = csr_matrix(X_train_sound)
X_val_sound_sparse = csr_matrix(X_val_sound)
X_test_sound_sparse = csr_matrix(X_test_sound)

# Combine sound and lyrics features
X_train_combined = hstack(
    [X_train_sound_sparse, X_train_lyrics],
    format="csr"
)

X_val_combined = hstack(
    [X_val_sound_sparse, X_val_lyrics],
    format="csr"
)

X_test_combined = hstack(
    [X_test_sound_sparse, X_test_lyrics],
    format="csr"
)

# Display combined dataset shapes
print("\nCombined feature shapes:")
print("Training:", X_train_combined.shape)
print("Validation:", X_val_combined.shape)
print("Testing:", X_test_combined.shape)

# Check matrix types
print("\nCombined matrix types:")
print("Training:", type(X_train_combined))
print("Validation:", type(X_val_combined))
print("Testing:", type(X_test_combined))

# Check the number of features
print("\nTotal features per song:", X_train_combined.shape[1])

print("\nSound + lyrics combination completed!")

# ==================================================
# STEP 50: CREATE COMBINED DATASET AND DATALOADER
# ==================================================

from torch.utils.data import Dataset, DataLoader

print("\nCreating combined model DataLoader...")

# Get activity labels for training songs
y_train_combined = df.loc[
    train_indices, activities
].to_numpy(dtype=np.float32)


# Create a custom dataset
class CombinedDataset(Dataset):

    def __init__(self, features, labels):

        self.features = features
        self.labels = labels

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self, index):

        # Convert only one song to a dense array
        song_features = (
            self.features[index]
            .toarray()
            .ravel()
        )

        # Get the song's activity labels
        song_labels = self.labels[index]

        # Convert to PyTorch tensors
        return (
            torch.from_numpy(song_features),
            torch.from_numpy(song_labels)
        )


# Create training dataset
combined_train_dataset = CombinedDataset(
    X_train_combined,
    y_train_combined
)

# Create DataLoader
combined_train_loader = DataLoader(
    combined_train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0
)

# Test one batch
batch_features, batch_labels = next(
    iter(combined_train_loader)
)

print("\nCombined training songs:", len(combined_train_dataset))

print("Number of batches:", len(combined_train_loader))

print("\nFirst batch shapes:")
print("Features:", batch_features.shape)
print("Labels:", batch_labels.shape)

print("\nCombined DataLoader created successfully!")

# ==================================================
# STEP 51: BUILD COMBINED NEURAL NETWORK
# ==================================================

import torch.nn as nn

print("\nBuilding combined Words + Sound neural network...")


# Create combined neural network
class CombinedModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input layer: 5008 features
            nn.Linear(5008, 128),
            nn.ReLU(),

            # Hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer: 5 activities
            nn.Linear(64, 5)

        )

    def forward(self, x):

        return self.network(x)


# Set random seed for reproducibility
torch.manual_seed(42)

# Create model
combined_model = CombinedModel()

# Display architecture
print("\nCombined Model Architecture:")
print(combined_model)

# Test model using first batch
with torch.no_grad():

    test_outputs = combined_model(batch_features)

print("\nTesting model with one batch...")

print("Input shape:", batch_features.shape)
print("Output shape:", test_outputs.shape)

print("\nCombined neural network created successfully!")

# ==================================================
# STEP 52: SET UP LOSS FUNCTION AND OPTIMIZER
# ==================================================

print("\nSetting up combined model training...")

# Count positive labels for each activity
positive_counts = y_train_combined.sum(axis=0)

# Count negative labels for each activity
negative_counts = (
    len(y_train_combined) - positive_counts
)

# Calculate class weights
combined_class_weights = (
    negative_counts / np.maximum(positive_counts, 1)
)

# Convert weights to PyTorch tensor
combined_weights_tensor = torch.tensor(
    combined_class_weights,
    dtype=torch.float32
)

# Create loss function
combined_loss_function = nn.BCEWithLogitsLoss(
    pos_weight=combined_weights_tensor
)

# Create optimizer
combined_optimizer = torch.optim.Adam(
    combined_model.parameters(),
    lr=0.001
)

# Display class weights
print("\nCombined Model Class Weights:")

for activity, weight in zip(
    activities,
    combined_class_weights
):
    print(activity, ":", round(weight, 2))

# Test loss function using first batch
test_loss = combined_loss_function(
    test_outputs,
    batch_labels
)

print("\nTest batch loss:", round(test_loss.item(), 4))

print("\nCombined model training setup completed!")

# ==================================================
# STEP 53: TRAIN COMBINED WORDS + SOUND MODEL
# ==================================================

print("\nTraining combined Words + Sound model...")

# Train for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):

    # Put model into training mode
    combined_model.train()

    total_loss = 0

    # Loop through training batches
    for batch_features, batch_labels in combined_train_loader:

        # Clear previous gradients
        combined_optimizer.zero_grad()

        # Make predictions
        predictions = combined_model(batch_features)

        # Calculate loss
        loss = combined_loss_function(
            predictions,
            batch_labels
        )

        # Calculate gradients
        loss.backward()

        # Update model weights
        combined_optimizer.step()

        # Track loss
        total_loss += loss.item()

    # Calculate average loss
    average_loss = total_loss / len(combined_train_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"| Average Training Loss: {average_loss:.4f}",
        flush=True
    )

print("\nCombined model training completed!")

# ==================================================
# STEP 54: EVALUATE COMBINED WORDS + SOUND MODEL
# ==================================================

from sklearn.metrics import f1_score, classification_report

print("\nEvaluating combined Words + Sound model...")

# Get correct activity labels for validation songs
y_val_combined = df.loc[
    val_indices, activities
].to_numpy(dtype=np.float32)

# Create validation dataset
combined_val_dataset = CombinedDataset(
    X_val_combined,
    y_val_combined
)

# Create validation DataLoader
combined_val_loader = DataLoader(
    combined_val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0
)

# Put model into evaluation mode
combined_model.eval()

# Store prediction probabilities
combined_val_probabilities = []

# Make predictions without training
with torch.no_grad():

    for batch_features, batch_labels in combined_val_loader:

        outputs = combined_model(batch_features)

        probabilities = torch.sigmoid(outputs)

        combined_val_probabilities.append(
            probabilities.numpy()
        )

# Combine predictions from all batches
combined_val_probabilities = np.concatenate(
    combined_val_probabilities,
    axis=0
)

# Apply initial threshold of 0.5
combined_val_predictions = (
    combined_val_probabilities >= 0.5
).astype(int)

# Calculate F1 scores
combined_micro_f1 = f1_score(
    y_val_combined,
    combined_val_predictions,
    average="micro",
    zero_division=0
)

combined_macro_f1 = f1_score(
    y_val_combined,
    combined_val_predictions,
    average="macro",
    zero_division=0
)

# Display results
print("\nCombined Micro F1:", round(combined_micro_f1, 4))
print("Combined Macro F1:", round(combined_macro_f1, 4))

print("\nPerformance by activity:")

print(
    classification_report(
        y_val_combined,
        combined_val_predictions,
        target_names=activities,
        zero_division=0
    )
)

print("\nCombined model evaluation completed!")

# ==================================================
# STEP 55: OPTIMIZE COMBINED MODEL THRESHOLDS
# ==================================================

print("\nOptimizing combined model thresholds...")

# Store the best threshold for each activity
combined_best_thresholds = []

# Test thresholds from 0.05 to 0.95
threshold_options = np.arange(0.05, 0.96, 0.05)

for i, activity in enumerate(activities):

    best_f1 = -1
    best_threshold = 0.5

    for threshold in threshold_options:

        # Make predictions using this threshold
        activity_predictions = (
            combined_val_probabilities[:, i] >= threshold
        ).astype(int)

        # Calculate F1 score
        score = f1_score(
            y_val_combined[:, i],
            activity_predictions,
            zero_division=0
        )

        # Keep the threshold with the highest F1
        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    combined_best_thresholds.append(best_threshold)

    print(
        activity,
        "| Best threshold:", round(best_threshold, 2),
        "| F1:", round(best_f1, 4)
    )

# Apply optimized thresholds
combined_optimized_predictions = np.zeros_like(
    y_val_combined,
    dtype=int
)

for i, threshold in enumerate(combined_best_thresholds):

    combined_optimized_predictions[:, i] = (
        combined_val_probabilities[:, i] >= threshold
    ).astype(int)

# Calculate optimized scores
combined_optimized_micro = f1_score(
    y_val_combined,
    combined_optimized_predictions,
    average="micro",
    zero_division=0
)

combined_optimized_macro = f1_score(
    y_val_combined,
    combined_optimized_predictions,
    average="macro",
    zero_division=0
)

print("\nOptimized Combined Micro F1:",
      round(combined_optimized_micro, 4))

print("Optimized Combined Macro F1:",
      round(combined_optimized_macro, 4))

print("\nCombined threshold optimization completed!")

# ==================================================
# STEP 56: SAVE COMBINED WORDS + SOUND MODEL
# ==================================================

print("\nSaving combined Words + Sound model...")

# Save trained neural network
torch.save(
    combined_model.state_dict(),
    "models/combined_model_grouped.pth"
)

# Save optimized thresholds
np.save(
    "models/combined_thresholds_grouped.npy",
    np.array(combined_best_thresholds)
)

# Save model information
joblib.dump(
    {
        "epochs": 5,
        "input_features": 5008,
        "sound_features": sound_features,
        "lyrics_features": 5000,
        "activities": activities,
        "micro_f1": combined_optimized_micro,
        "macro_f1": combined_optimized_macro
    },
    "models/combined_metadata_grouped.pkl"
)

print("\nCombined model saved successfully!")

print("Micro F1:", round(combined_optimized_micro, 4))
print("Macro F1:", round(combined_optimized_macro, 4))