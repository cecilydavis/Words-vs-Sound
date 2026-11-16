# ==================================================
# WORDS VS. SOUND
# Sound-Only Neural Network
# ==================================================


# ==================================================
# STEP 1: IMPORT LIBRARIES
# ==================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Limit PyTorch to one CPU thread
torch.set_num_threads(1)

print("PyTorch CPU threads:", torch.get_num_threads())


# ==================================================
# STEP 2: LOAD AND EXPLORE THE DATASET
# ==================================================

df = pd.read_csv("data/spotify_dataset.csv")

print("\nTotal songs:", len(df))

print("\nMissing values:")
print(df.isnull().sum())


# Define our five activity categories
activities = [
    "Good for Party",
    "Good for Work/Study",
    "Good for Relaxation/Meditation",
    "Good for Exercise",
    "Good for Driving"
]

# Count songs in each activity
print("\nSongs per activity:")

for activity in activities:
    print(activity)
    print(df[activity].value_counts())
    print()

# Check activity label data types
print("\nActivity label data types:")
print(df[activities].dtypes)

# Count how many activities each song belongs to
df["activity_count"] = df[activities].sum(axis=1)

print("\nNumber of activities per song:")
print(df["activity_count"].value_counts().sort_index())


# ==================================================
# STEP 3: SELECT SOUND FEATURES
# ==================================================

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

# Separate inputs and outputs
X = df[sound_features]
y = df[activities]

print("\nSound features:")
print(X.head())

print("\nActivity labels:")
print(y.head())

print("\nInput shape:", X.shape)
print("Output shape:", y.shape)


# ==================================================
# STEP 4: CLEAN SOUND FEATURES
# ==================================================

# Remove "db" from loudness values
df["Loudness (db)"] = (
    df["Loudness (db)"]
    .astype(str)
    .str.replace("db", "", regex=False)
)

# Convert sound features into numbers
for feature in sound_features:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

print("\nMissing values in sound features:")
print(df[sound_features].isnull().sum())

print("\nSound feature data types:")
print(df[sound_features].dtypes)

print("\nCleaned sound features:")
print(df[sound_features].head())


# ==================================================
# STEP 5: REMOVE DUPLICATE ROWS
# ==================================================

df = df.drop_duplicates().reset_index(drop=True)

print("\nSongs after removing duplicates:", len(df))

print("Remaining duplicate rows:", df.duplicated().sum())

# Check repeated artist/song combinations
duplicates = df.duplicated(
    subset=["Artist(s)", "song"]
).sum()

print("\nDuplicate artist/song combinations:", duplicates)


# ==================================================
# STEP 6: SPLIT DATA INTO TRAINING, VALIDATION, TESTING
# ==================================================

# Recreate inputs and outputs using cleaned data
X = df[sound_features]
y = df[activities]

# First split: 70% training, 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Second split: 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\nTraining songs:", len(X_train))
print("Validation songs:", len(X_val))
print("Testing songs:", len(X_test))

print("\nTraining activity counts:")
print(y_train.sum())

print("\nValidation activity counts:")
print(y_val.sum())

print("\nTesting activity counts:")
print(y_test.sum())


# ==================================================
# STEP 7: STANDARDIZE SOUND FEATURES
# ==================================================

scaler = StandardScaler()

# Learn scaling values from training data only
X_train_scaled = scaler.fit_transform(X_train)

# Apply same scaling to validation and testing data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nStandardized training features:")
print(X_train_scaled[:5])

print("\nFeature averages:")
print(X_train_scaled.mean(axis=0))

print("\nFeature standard deviations:")
print(X_train_scaled.std(axis=0))


# ==================================================
# STEP 8: CONVERT DATA INTO PYTORCH TENSORS
# ==================================================

# Convert sound features into tensors
X_train_tensor = torch.from_numpy(
    np.ascontiguousarray(
        X_train_scaled,
        dtype=np.float32
    )
)

X_val_tensor = torch.from_numpy(
    np.ascontiguousarray(
        X_val_scaled,
        dtype=np.float32
    )
)

X_test_tensor = torch.from_numpy(
    np.ascontiguousarray(
        X_test_scaled,
        dtype=np.float32
    )
)

# Convert activity labels into tensors
y_train_tensor = torch.from_numpy(
    np.ascontiguousarray(
        y_train.to_numpy(),
        dtype=np.float32
    )
)

y_val_tensor = torch.from_numpy(
    np.ascontiguousarray(
        y_val.to_numpy(),
        dtype=np.float32
    )
)

y_test_tensor = torch.from_numpy(
    np.ascontiguousarray(
        y_test.to_numpy(),
        dtype=np.float32
    )
)

print("\nTraining input:", X_train_tensor.shape)
print("Training output:", y_train_tensor.shape)

print("\nValidation input:", X_val_tensor.shape)
print("Validation output:", y_val_tensor.shape)

print("\nTesting input:", X_test_tensor.shape)
print("Testing output:", y_test_tensor.shape)


# ==================================================
# STEP 9: EXPLORE CORRELATIONS
# ==================================================

correlations = df[sound_features + activities].corr()

print("\nSound feature correlations with activities:")
print(
    correlations.loc[
        sound_features,
        activities
    ].round(2)
)


# ==================================================
# STEP 10: BUILD SOUND-ONLY NEURAL NETWORK
# ==================================================

class SoundModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, x):
        return self.network(x)


# Create the model
sound_model = SoundModel()

print("\nSound-Only Neural Network:")
print(sound_model)


# ==================================================
# STEP 11: CALCULATE CLASS WEIGHTS
# ==================================================

# Use NumPy to avoid previous PyTorch crashes
positive_counts = y_train.to_numpy().sum(axis=0)

negative_counts = len(y_train) - positive_counts

class_weights = negative_counts / positive_counts

print("\nActivity class weights:")

for activity, weight in zip(activities, class_weights):
    print(activity, ":", round(float(weight), 2))


# ==================================================
# STEP 12: CREATE LOSS FUNCTION AND OPTIMIZER
# ==================================================

# Convert class weights into a tensor
weights_tensor = torch.from_numpy(
    np.array(
        class_weights,
        dtype=np.float32
    )
)

# Weighted loss function
loss_function = nn.BCEWithLogitsLoss(
    pos_weight=weights_tensor
)

# Adam optimizer
optimizer = torch.optim.Adam(
    sound_model.parameters(),
    lr=0.001
)

print("\nLoss function created successfully!")
print("Optimizer created successfully!")


# ==================================================
# STEP 13: TEST THE MODEL WITH FIVE SONGS
# ==================================================

print("\nTesting model with 5 songs...")

# Select five songs
sample_songs = X_train_tensor[:5].clone().contiguous()

# Select their activity labels
sample_labels = y_train_tensor[:5].clone().contiguous()

# Display tensor information
print("Song tensor shape:", sample_songs.shape)
print("Label tensor shape:", sample_labels.shape)

print("Song tensor dtype:", sample_songs.dtype)
print("Label tensor dtype:", sample_labels.dtype)

# Generate predictions
sample_predictions = sound_model(sample_songs)

print("\nPredictions created successfully!")

# Create a temporary unweighted loss function
test_loss_function = nn.BCEWithLogitsLoss()

print("Calculating unweighted loss...")

# Calculate loss
test_loss = test_loss_function(
    sample_predictions,
    sample_labels
)

print("\nUnweighted loss calculated successfully!")
print("Loss:", test_loss.item())


# ==================================================
# END OF CURRENT PROJECT PROGRESS
# ==================================================

print("\nCurrent testing step completed!")

# ==================================================
# STEP 16: TEST ONE TRAINING STEP
# ==================================================

print("\nTesting one training step...")

# Clear previous gradients
optimizer.zero_grad()

# Make predictions using five songs
training_predictions = sound_model(sample_songs)

# Calculate weighted loss
training_loss = loss_function(
    training_predictions,
    sample_labels
)

print("Loss before updating:", training_loss.item())

# Calculate gradients
training_loss.backward()

print("Backward pass completed successfully!")

# Update the model's weights
optimizer.step()

print("Model weights updated successfully!")

# Check the loss after updating
with torch.no_grad():
    updated_predictions = sound_model(sample_songs)

    updated_loss = loss_function(
        updated_predictions,
        sample_labels
    )

print("Loss after updating:", updated_loss.item())

# ==================================================
# STEP 17: PREPARE TRAINING BATCHES
# ==================================================

from torch.utils.data import TensorDataset, DataLoader

# Combine training inputs and labels
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

# Create batches of 256 songs
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0
)

print("\nTraining batches created successfully!")

print("Total training songs:", len(train_dataset))
print("Number of batches:", len(train_loader))

# Test the first batch
first_batch = next(iter(train_loader))

batch_songs, batch_labels = first_batch

print("\nFirst batch input shape:", batch_songs.shape)
print("First batch label shape:", batch_labels.shape)

# ==================================================
# STEP 18: TRAIN THE SOUND-ONLY NEURAL NETWORK
# ==================================================

print("\nStarting sound-only model training...")

# Create a fresh model for actual training
# This removes the effects of our earlier five-song test
torch.manual_seed(42)

sound_model = SoundModel()

# Create a fresh optimizer
optimizer = torch.optim.Adam(
    sound_model.parameters(),
    lr=0.001
)

# Put the model in training mode
sound_model.train()

# Track the total loss
total_loss = 0

# Go through every batch
for batch_songs, batch_labels in train_loader:

    # Clear previous gradients
    optimizer.zero_grad()

    # Make predictions
    predictions = sound_model(batch_songs)

    # Calculate weighted loss
    loss = loss_function(
        predictions,
        batch_labels
    )

    # Calculate gradients
    loss.backward()

    # Update model weights
    optimizer.step()

    # Record the loss
    total_loss += loss.item()

# Calculate average loss
average_loss = total_loss / len(train_loader)

print("\nTraining completed successfully!")
print("Average training loss:", round(average_loss, 4))

# ==================================================
# STEP 19: EVALUATE SOUND-ONLY MODEL
# ==================================================

from sklearn.metrics import classification_report, f1_score

print("\nEvaluating sound-only model...")

# Put model in evaluation mode
sound_model.eval()

# Make predictions without updating weights
with torch.no_grad():

    # Get raw prediction scores
    validation_outputs = sound_model(X_val_tensor)

    # Convert scores into probabilities
    validation_probabilities = torch.sigmoid(validation_outputs)

    # Convert probabilities into 0 or 1 predictions
    validation_predictions = (
        validation_probabilities >= 0.5
    ).int().numpy()

# Get actual activity labels
actual_labels = y_val.to_numpy()

# Calculate overall F1 scores
micro_f1 = f1_score(
    actual_labels,
    validation_predictions,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    actual_labels,
    validation_predictions,
    average="macro",
    zero_division=0
)

print("\nMicro F1 Score:", round(micro_f1, 4))
print("Macro F1 Score:", round(macro_f1, 4))

# Show performance for each activity
print("\nPerformance by activity:")

print(
    classification_report(
        actual_labels,
        validation_predictions,
        target_names=activities,
        zero_division=0
    )
)

# ==================================================
# STEP 20: TEST DIFFERENT PREDICTION THRESHOLDS
# ==================================================

print("\nTesting different prediction thresholds...")

# Convert probabilities to NumPy
probabilities = validation_probabilities.numpy()

# Test several thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:

    # Convert probabilities into predictions
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Calculate F1 scores
    micro = f1_score(
        actual_labels,
        predictions,
        average="micro",
        zero_division=0
    )

    macro = f1_score(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        "Threshold:", threshold,
        "| Micro F1:", round(micro, 4),
        "| Macro F1:", round(macro, 4)
    )

# ==================================================
# STEP 21: FIND BEST THRESHOLD FOR EACH ACTIVITY
# ==================================================

print("\nFinding the best threshold for each activity...")

best_thresholds = []

# Test thresholds from 0.50 to 0.95
threshold_options = np.arange(0.50, 0.96, 0.05)

for i, activity in enumerate(activities):

    best_f1 = -1
    best_threshold = 0.5

    for threshold in threshold_options:

        # Get predictions for this activity
        activity_predictions = (
            probabilities[:, i] >= threshold
        ).astype(int)

        # Calculate F1 score
        score = f1_score(
            actual_labels[:, i],
            activity_predictions,
            zero_division=0
        )

        # Save the best result
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    best_thresholds.append(best_threshold)

    print(
        activity,
        "| Best threshold:", round(best_threshold, 2),
        "| F1:", round(best_f1, 4)
    )

# Apply the best threshold to each activity
optimized_predictions = np.zeros_like(actual_labels)

for i, threshold in enumerate(best_thresholds):
    optimized_predictions[:, i] = (
        probabilities[:, i] >= threshold
    ).astype(int)

# Calculate overall performance
optimized_micro = f1_score(
    actual_labels,
    optimized_predictions,
    average="micro",
    zero_division=0
)

optimized_macro = f1_score(
    actual_labels,
    optimized_predictions,
    average="macro",
    zero_division=0
)

print("\nOptimized Micro F1:", round(optimized_micro, 4))
print("Optimized Macro F1:", round(optimized_macro, 4))

# ==================================================
# STEP 22: SAVE SOUND-ONLY MODEL
# ==================================================

import os
import joblib

# Create a folder for saved models
os.makedirs("models", exist_ok=True)

# Save the trained neural network
torch.save(
    sound_model.state_dict(),
    "models/sound_model.pth"
)

# Save the optimized thresholds
np.save(
    "models/sound_thresholds.npy",
    np.array(best_thresholds)
)

# Save the scaler
joblib.dump(
    scaler,
    "models/sound_scaler.pkl"
)

# Save feature names and activity categories
joblib.dump(
    {
        "sound_features": sound_features,
        "activities": activities
    },
    "models/sound_metadata.pkl"
)

print("\nSound-only model saved successfully!")

# ==================================================
# STEP 23: SAVE DATASET SPLITS
# ==================================================

print("\nSaving dataset splits...")

# Save the original row indices for each dataset
np.save(
    "models/train_indices.npy",
    X_train.index.to_numpy()
)

np.save(
    "models/val_indices.npy",
    X_val.index.to_numpy()
)

np.save(
    "models/test_indices.npy",
    X_test.index.to_numpy()
)

print("\nDataset splits saved successfully!")

print("Training indices:", len(X_train.index))
print("Validation indices:", len(X_val.index))
print("Testing indices:", len(X_test.index))

# ==================================================
# STEP 24A: INSPECT LYRICS DATA
# ==================================================

print("\nInspecting lyrics data...")

# Check how many songs have missing lyrics
missing_lyrics = df["text"].isna().sum()

print("\nTotal songs:", len(df))
print("Songs with missing lyrics:", missing_lyrics)

# Convert lyrics to strings for inspection
lyrics = df["text"].fillna("").astype(str)

# Check for empty lyrics
empty_lyrics = (lyrics.str.strip() == "").sum()

print("Songs with empty lyrics:", empty_lyrics)

# Calculate lyric lengths
lyric_lengths = lyrics.str.split().str.len()

print("\nAverage words per song:", round(lyric_lengths.mean(), 2))
print("Longest song (words):", lyric_lengths.max())
print("Shortest song (words):", lyric_lengths.min())

# Display a short example
print("\nExample lyrics:")
print(lyrics.iloc[0][:500])

# ==================================================
# STEP 24B: CHECK FOR DUPLICATE SONGS ACROSS SPLITS
# ==================================================

print("\nChecking for duplicate songs across dataset splits...")

# Load the saved row indices
train_indices = np.load("models/train_indices.npy")
val_indices = np.load("models/val_indices.npy")
test_indices = np.load("models/test_indices.npy")

# Verify that indices are unique and cover the cleaned dataset
all_indices = np.concatenate([
    train_indices,
    val_indices,
    test_indices
])

print("Unique saved indices:", len(np.unique(all_indices)))
print("Total cleaned songs:", len(df))

# Identify songs using artist and song title
song_keys = (
    df["Artist(s)"].fillna("").astype(str).str.strip().str.lower()
    + " | "
    + df["song"].fillna("").astype(str).str.strip().str.lower()
)

# Get the songs in each split
train_songs = set(song_keys.loc[train_indices])
val_songs = set(song_keys.loc[val_indices])
test_songs = set(song_keys.loc[test_indices])

# Check for overlapping songs
train_val_overlap = len(train_songs & val_songs)
train_test_overlap = len(train_songs & test_songs)
val_test_overlap = len(val_songs & test_songs)

print("\nSongs appearing in multiple splits:")
print("Training and validation:", train_val_overlap)
print("Training and testing:", train_test_overlap)
print("Validation and testing:", val_test_overlap)

print("\nDuplicate song check completed!")

# ==================================================
# STEP 25A: CREATE GROUP-BASED DATASET SPLITS
# ==================================================

from sklearn.model_selection import GroupShuffleSplit

print("\nCreating new dataset splits without song overlap...")

# Identify each song using its artist and title
song_groups = (
    df["Artist(s)"].fillna("").astype(str).str.strip().str.lower()
    + " | "
    + df["song"].fillna("").astype(str).str.strip().str.lower()
)

# First split: 70% training, 30% remaining
first_split = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=42
)

train_positions, remaining_positions = next(
    first_split.split(df, groups=song_groups)
)

# Second split: divide remaining songs into validation and testing
remaining_groups = song_groups.iloc[remaining_positions]

second_split = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=42
)

val_positions, test_positions = next(
    second_split.split(
        remaining_positions,
        groups=remaining_groups
    )
)

# Convert positions into original DataFrame indices
new_train_indices = df.index[train_positions].to_numpy()

new_val_indices = df.index[
    remaining_positions[val_positions]
].to_numpy()

new_test_indices = df.index[
    remaining_positions[test_positions]
].to_numpy()

# Check the number of songs
print("\nNew dataset sizes:")
print("Training:", len(new_train_indices))
print("Validation:", len(new_val_indices))
print("Testing:", len(new_test_indices))

# Verify that no song identities overlap
new_train_songs = set(song_groups.loc[new_train_indices])
new_val_songs = set(song_groups.loc[new_val_indices])
new_test_songs = set(song_groups.loc[new_test_indices])

print("\nChecking for overlapping songs:")
print("Training and validation:", len(new_train_songs & new_val_songs))
print("Training and testing:", len(new_train_songs & new_test_songs))
print("Validation and testing:", len(new_val_songs & new_test_songs))

# Verify that all rows were assigned exactly once
all_new_indices = np.concatenate([
    new_train_indices,
    new_val_indices,
    new_test_indices
])

print("\nTotal assigned rows:", len(all_new_indices))
print("Unique assigned rows:", len(np.unique(all_new_indices)))

print("\nGroup-based splitting completed!")

# ==================================================
# STEP 25B: SAVE CORRECTED DATASET SPLITS
# ==================================================

print("\nSaving corrected dataset splits...")

# Save the corrected training indices
np.save(
    "models/train_indices_grouped.npy",
    new_train_indices
)

# Save the corrected validation indices
np.save(
    "models/val_indices_grouped.npy",
    new_val_indices
)

# Save the corrected testing indices
np.save(
    "models/test_indices_grouped.npy",
    new_test_indices
)

# Verify the saved files
saved_train = np.load("models/train_indices_grouped.npy")
saved_val = np.load("models/val_indices_grouped.npy")
saved_test = np.load("models/test_indices_grouped.npy")

print("\nCorrected dataset splits saved successfully!")

print("Saved training songs:", len(saved_train))
print("Saved validation songs:", len(saved_val))
print("Saved testing songs:", len(saved_test))

print("\nAll corrected split files are ready!")

# ==================================================
# STEP 26: LOAD CORRECTED DATASETS
# ==================================================

print("\nLoading corrected datasets...")

# Load corrected indices
train_indices = np.load("models/train_indices_grouped.npy")
val_indices = np.load("models/val_indices_grouped.npy")
test_indices = np.load("models/test_indices_grouped.npy")

# Select sound features for each dataset
X_train_grouped = df.loc[train_indices, sound_features]
X_val_grouped = df.loc[val_indices, sound_features]
X_test_grouped = df.loc[test_indices, sound_features]

# Select activity labels for each dataset
y_train_grouped = df.loc[train_indices, activities]
y_val_grouped = df.loc[val_indices, activities]
y_test_grouped = df.loc[test_indices, activities]

# Verify dataset shapes
print("\nCorrected training features:", X_train_grouped.shape)
print("Corrected training labels:", y_train_grouped.shape)

print("\nCorrected validation features:", X_val_grouped.shape)
print("Corrected validation labels:", y_val_grouped.shape)

print("\nCorrected testing features:", X_test_grouped.shape)
print("Corrected testing labels:", y_test_grouped.shape)

print("\nCorrected datasets loaded successfully!")

# ==================================================
# STEP 27: STANDARDIZE CORRECTED SOUND DATA
# ==================================================

print("\nStandardizing corrected sound data...")

# Create a new scaler
grouped_scaler = StandardScaler()

# Fit the scaler using ONLY the corrected training data
X_train_grouped_scaled = grouped_scaler.fit_transform(
    X_train_grouped
)

# Apply the same scaling to validation and testing data
X_val_grouped_scaled = grouped_scaler.transform(
    X_val_grouped
)

X_test_grouped_scaled = grouped_scaler.transform(
    X_test_grouped
)

# Verify the results
print("\nScaled training shape:", X_train_grouped_scaled.shape)
print("Scaled validation shape:", X_val_grouped_scaled.shape)
print("Scaled testing shape:", X_test_grouped_scaled.shape)

print("\nAverage training feature values:")
print(np.round(X_train_grouped_scaled.mean(axis=0), 4))

print("\nStandardized corrected sound data successfully!")

# ==================================================
# STEP 28: CONVERT CORRECTED DATA TO PYTORCH TENSORS
# ==================================================

print("\nConverting corrected datasets to PyTorch tensors...")

# Convert training data
X_train_grouped_tensor = torch.from_numpy(
    np.ascontiguousarray(X_train_grouped_scaled, dtype=np.float32)
)

y_train_grouped_tensor = torch.from_numpy(
    np.ascontiguousarray(y_train_grouped, dtype=np.float32)
)

# Convert validation data
X_val_grouped_tensor = torch.from_numpy(
    np.ascontiguousarray(X_val_grouped_scaled, dtype=np.float32)
)

y_val_grouped_tensor = torch.from_numpy(
    np.ascontiguousarray(y_val_grouped, dtype=np.float32)
)

# Convert testing data
X_test_grouped_tensor = torch.from_numpy(
    np.ascontiguousarray(X_test_grouped_scaled, dtype=np.float32)
)

y_test_grouped_tensor = torch.from_numpy(
    np.ascontiguousarray(y_test_grouped, dtype=np.float32)
)

# Verify tensor shapes
print("\nTraining input:", X_train_grouped_tensor.shape)
print("Training labels:", y_train_grouped_tensor.shape)

print("\nValidation input:", X_val_grouped_tensor.shape)
print("Validation labels:", y_val_grouped_tensor.shape)

print("\nTesting input:", X_test_grouped_tensor.shape)
print("Testing labels:", y_test_grouped_tensor.shape)

print("\nCorrected tensors created successfully!")

# ==================================================
# STEP 29: CREATE CORRECTED TRAINING BATCHES
# ==================================================

print("\nCreating corrected training batches...")

# Combine sound features and labels
grouped_train_dataset = TensorDataset(
    X_train_grouped_tensor,
    y_train_grouped_tensor
)

# Create batches of 256 songs
grouped_train_loader = DataLoader(
    grouped_train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0
)

# Verify the results
print("\nTotal training songs:", len(grouped_train_dataset))
print("Number of training batches:", len(grouped_train_loader))

# Check the first batch
first_grouped_batch = next(iter(grouped_train_loader))

batch_songs, batch_labels = first_grouped_batch

print("\nFirst batch input shape:", batch_songs.shape)
print("First batch label shape:", batch_labels.shape)

print("\nCorrected training batches created successfully!")

# ==================================================
# STEP 30: RETRAIN SOUND-ONLY MODEL
# ==================================================

print("\nRetraining sound-only model with corrected data...")

# Set random seed for reproducibility
torch.manual_seed(42)

# Create a fresh neural network
grouped_sound_model = SoundModel()

# Calculate new class weights using corrected training labels
positive_counts = y_train_grouped.to_numpy().sum(axis=0)
negative_counts = len(y_train_grouped) - positive_counts

grouped_weights = negative_counts / np.maximum(positive_counts, 1)

grouped_weights_tensor = torch.tensor(
    grouped_weights,
    dtype=torch.float32
)

# Create weighted loss function
grouped_loss_function = nn.BCEWithLogitsLoss(
    pos_weight=grouped_weights_tensor
)

# Create optimizer
grouped_optimizer = torch.optim.Adam(
    grouped_sound_model.parameters(),
    lr=0.001
)

# Train for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):

    grouped_sound_model.train()

    total_loss = 0

    for batch_songs, batch_labels in grouped_train_loader:

        # Clear previous gradients
        grouped_optimizer.zero_grad()

        # Make predictions
        predictions = grouped_sound_model(batch_songs)

        # Calculate loss
        loss = grouped_loss_function(
            predictions,
            batch_labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        grouped_optimizer.step()

        # Track loss
        total_loss += loss.item()

    average_loss = total_loss / len(grouped_train_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"| Average Training Loss: {average_loss:.4f}"
    )

print("\nCorrected sound-only model training completed!")

# ==================================================
# STEP 31: EVALUATE CORRECTED SOUND-ONLY MODEL
# ==================================================

print("\nEvaluating corrected sound-only model...")

# Put the model in evaluation mode
grouped_sound_model.eval()

# Make predictions without updating model weights
with torch.no_grad():

    # Get raw model outputs
    grouped_validation_outputs = grouped_sound_model(
        X_val_grouped_tensor
    )

    # Convert outputs into probabilities
    grouped_validation_probabilities = torch.sigmoid(
        grouped_validation_outputs
    ).numpy()

# Get actual validation labels
grouped_actual_labels = y_val_grouped.to_numpy()

# Start with a threshold of 0.5
grouped_predictions = (
    grouped_validation_probabilities >= 0.5
).astype(int)

# Calculate Micro F1
grouped_micro_f1 = f1_score(
    grouped_actual_labels,
    grouped_predictions,
    average="micro",
    zero_division=0
)

# Calculate Macro F1
grouped_macro_f1 = f1_score(
    grouped_actual_labels,
    grouped_predictions,
    average="macro",
    zero_division=0
)

print("\nCorrected Micro F1:", round(grouped_micro_f1, 4))
print("Corrected Macro F1:", round(grouped_macro_f1, 4))

# Show performance for each activity
print("\nPerformance by activity:")

print(
    classification_report(
        grouped_actual_labels,
        grouped_predictions,
        target_names=activities,
        zero_division=0
    )
)

print("\nCorrected sound-only evaluation completed!")

# ==================================================
# STEP 32: OPTIMIZE CORRECTED MODEL THRESHOLDS
# ==================================================

print("\nOptimizing corrected model thresholds...")

grouped_best_thresholds = []

# Test thresholds from 0.50 to 0.95
threshold_options = np.arange(0.50, 0.96, 0.05)

for i, activity in enumerate(activities):

    best_f1 = -1
    best_threshold = 0.5

    for threshold in threshold_options:

        # Make predictions for this activity
        activity_predictions = (
            grouped_validation_probabilities[:, i] >= threshold
        ).astype(int)

        # Calculate F1 score
        score = f1_score(
            grouped_actual_labels[:, i],
            activity_predictions,
            zero_division=0
        )

        # Save the best result
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    grouped_best_thresholds.append(best_threshold)

    print(
        activity,
        "| Best threshold:", round(best_threshold, 2),
        "| F1:", round(best_f1, 4)
    )

# Apply the optimized thresholds
grouped_optimized_predictions = np.zeros_like(
    grouped_actual_labels
)

for i, threshold in enumerate(grouped_best_thresholds):

    grouped_optimized_predictions[:, i] = (
        grouped_validation_probabilities[:, i] >= threshold
    ).astype(int)

# Calculate overall scores
grouped_optimized_micro = f1_score(
    grouped_actual_labels,
    grouped_optimized_predictions,
    average="micro",
    zero_division=0
)

grouped_optimized_macro = f1_score(
    grouped_actual_labels,
    grouped_optimized_predictions,
    average="macro",
    zero_division=0
)

print("\nOptimized Micro F1:", round(grouped_optimized_micro, 4))
print("Optimized Macro F1:", round(grouped_optimized_macro, 4))

print("\nCorrected threshold optimization completed!")

# ==================================================
# STEP 33: SAVE CORRECTED SOUND-ONLY MODEL
# ==================================================

print("\nSaving corrected sound-only model...")

# Save the corrected neural network
torch.save(
    grouped_sound_model.state_dict(),
    "models/sound_model_grouped.pth"
)

# Save optimized thresholds
np.save(
    "models/sound_thresholds_grouped.npy",
    np.array(grouped_best_thresholds)
)

# Save the corrected scaler
joblib.dump(
    grouped_scaler,
    "models/sound_scaler_grouped.pkl"
)

# Save model information
joblib.dump(
    {
        "sound_features": sound_features,
        "activities": activities,
        "epochs": 5,
        "micro_f1": grouped_optimized_micro,
        "macro_f1": grouped_optimized_macro
    },
    "models/sound_metadata_grouped.pkl"
)

print("\nCorrected sound-only model saved successfully!")

print("Micro F1:", round(grouped_optimized_micro, 4))
print("Macro F1:", round(grouped_optimized_macro, 4))