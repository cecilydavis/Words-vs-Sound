# ==================================================
# STEP 57: SET UP FINAL MODEL COMPARISON
# ==================================================

import os
import pandas as pd
import numpy as np
import torch
import joblib

# Prevent PyTorch crashes on Mac
torch.set_num_threads(1)

print("Starting final Words vs. Sound comparison...")

# Load dataset
df = pd.read_csv("data/spotify_dataset.csv")

# Remove exact duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Load the untouched testing indices
test_indices = np.load(
    "models/test_indices_grouped.npy"
)

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

# List all files needed for final testing
required_files = [
    "models/sound_model_grouped.pth",
    "models/sound_thresholds_grouped.npy",
    "models/sound_scaler_grouped.pkl",
    "models/lyrics_model_5epochs.pth",
    "models/lyrics_thresholds_5epochs.npy",
    "models/lyrics_tfidf.pkl",
    "models/combined_model_grouped.pth",
    "models/combined_thresholds_grouped.npy"
]

# Check that all saved files exist
print("\nChecking saved model files...")

all_files_found = True

for filename in required_files:

    if os.path.exists(filename):
        print("FOUND:", filename)

    else:
        print("MISSING:", filename)
        all_files_found = False

# Display testing dataset information
print("\nFinal testing dataset:")
print("Testing songs:", len(test_indices))
print("Sound features:", len(sound_features))
print("Lyrics features: 5000")
print("Activity labels:", len(activities))

if all_files_found:
    print("\nAll saved model files are ready!")
else:
    print("\nSome saved model files are missing.")

print("\nFinal comparison setup completed!")

# ==================================================
# STEP 58: LOAD ALL THREE TRAINED NEURAL NETWORKS
# ==================================================

import torch.nn as nn

print("\nLoading our three trained neural networks...")


# --------------------------------------------------
# 1. SOUND-ONLY MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# 2. LYRICS-ONLY MODEL
# --------------------------------------------------

class LyricsModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# 3. COMBINED WORDS + SOUND MODEL
# --------------------------------------------------

class CombinedModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5008, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# LOAD SAVED MODEL WEIGHTS
# --------------------------------------------------

# Sound-only
sound_model = SoundModel()

sound_model.load_state_dict(
    torch.load(
        "models/sound_model_grouped.pth",
        map_location="cpu",
        weights_only=True
    )
)

sound_model.eval()

print("Sound-only model loaded successfully!")


# Lyrics-only
lyrics_model = LyricsModel()

lyrics_model.load_state_dict(
    torch.load(
        "models/lyrics_model_5epochs.pth",
        map_location="cpu",
        weights_only=True
    )
)

lyrics_model.eval()

print("Lyrics-only model loaded successfully!")


# Combined
combined_model = CombinedModel()

combined_model.load_state_dict(
    torch.load(
        "models/combined_model_grouped.pth",
        map_location="cpu",
        weights_only=True
    )
)

combined_model.eval()

print("Combined model loaded successfully!")


# --------------------------------------------------
# LOAD OPTIMIZED THRESHOLDS
# --------------------------------------------------

sound_thresholds = np.load(
    "models/sound_thresholds_grouped.npy"
)

lyrics_thresholds = np.load(
    "models/lyrics_thresholds_5epochs.npy"
)

combined_thresholds = np.load(
    "models/combined_thresholds_grouped.npy"
)

print("\nOptimized thresholds loaded successfully!")

print("\nSound thresholds:", sound_thresholds)
print("Lyrics thresholds:", lyrics_thresholds)
print("Combined thresholds:", combined_thresholds)

print("\nAll three models are ready for final testing!")

# ==================================================
# STEP 59: PREPARE FINAL TESTING FEATURES
# ==================================================

from scipy.sparse import csr_matrix, hstack

print("\nPreparing final testing features...")


# --------------------------------------------------
# 1. GET TESTING SONGS
# --------------------------------------------------

test_df = df.loc[test_indices].copy()

print("\nTesting songs:", len(test_df))


# --------------------------------------------------
# 2. PREPARE SOUND FEATURES
# --------------------------------------------------

# Remove "db" from loudness values
test_df["Loudness (db)"] = (
    test_df["Loudness (db)"]
    .astype(str)
    .str.replace("db", "", regex=False)
    .str.strip()
)

# Convert sound features to numbers
for feature in sound_features:

    test_df[feature] = pd.to_numeric(
        test_df[feature],
        errors="coerce"
    )

# Check for missing sound values
print(
    "Missing sound values:",
    test_df[sound_features].isna().sum().sum()
)

# Load the scaler fitted on training data
sound_scaler = joblib.load(
    "models/sound_scaler_grouped.pkl"
)

# Scale testing sound features
X_test_sound = sound_scaler.transform(
    test_df[sound_features]
).astype(np.float32)

print("\nSound testing shape:", X_test_sound.shape)


# --------------------------------------------------
# 3. PREPARE LYRICS FEATURES
# --------------------------------------------------

# Load the TF-IDF vectorizer fitted on training data
lyrics_tfidf = joblib.load(
    "models/lyrics_tfidf.pkl"
)

# Convert lyrics into numerical features
X_test_lyrics = lyrics_tfidf.transform(
    test_df["text"].fillna("")
)

print("Lyrics testing shape:", X_test_lyrics.shape)


# --------------------------------------------------
# 4. PREPARE COMBINED FEATURES
# --------------------------------------------------

# Convert sound features into a sparse matrix
X_test_sound_sparse = csr_matrix(
    X_test_sound
)

# Combine sound and lyrics features
X_test_combined = hstack(
    [
        X_test_sound_sparse,
        X_test_lyrics
    ],
    format="csr"
).astype(np.float32)

print("Combined testing shape:", X_test_combined.shape)


# --------------------------------------------------
# 5. PREPARE CORRECT ACTIVITY LABELS
# --------------------------------------------------

y_test = test_df[activities].to_numpy(
    dtype=np.float32
)

print("Testing labels shape:", y_test.shape)


# --------------------------------------------------
# 6. VERIFY FEATURE TYPES
# --------------------------------------------------

print("\nFeature data types:")

print("Sound:", X_test_sound.dtype)
print("Lyrics:", X_test_lyrics.dtype)
print("Combined:", X_test_combined.dtype)

print("\nFinal testing features prepared successfully!")

# ==================================================
# STEP 60: GENERATE FINAL TESTING PREDICTIONS
# ==================================================

print("\nGenerating predictions from all three models...")


# --------------------------------------------------
# FUNCTION TO MAKE PREDICTIONS IN BATCHES
# --------------------------------------------------

def get_model_predictions(model, features, thresholds):

    model.eval()

    all_probabilities = []

    batch_size = 256

    with torch.no_grad():

        for start in range(0, features.shape[0], batch_size):

            end = start + batch_size

            # Get one batch of songs
            batch = features[start:end]

            # Convert sparse matrices to dense arrays
            if hasattr(batch, "toarray"):
                batch = batch.toarray()

            # Convert features into PyTorch tensors
            batch_tensor = torch.from_numpy(
                np.asarray(batch, dtype=np.float32)
            )

            # Make predictions
            outputs = model(batch_tensor)

            # Convert outputs into probabilities
            probabilities = torch.sigmoid(outputs)

            all_probabilities.append(
                probabilities.numpy()
            )

    # Combine predictions from every batch
    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    # Apply the saved optimized thresholds
    predictions = (
        all_probabilities >= thresholds
    ).astype(int)

    return predictions


# --------------------------------------------------
# SOUND-ONLY PREDICTIONS
# --------------------------------------------------

sound_predictions = get_model_predictions(
    sound_model,
    X_test_sound,
    sound_thresholds
)

print("Sound-only predictions completed!")


# --------------------------------------------------
# LYRICS-ONLY PREDICTIONS
# --------------------------------------------------

lyrics_predictions = get_model_predictions(
    lyrics_model,
    X_test_lyrics,
    lyrics_thresholds
)

print("Lyrics-only predictions completed!")


# --------------------------------------------------
# COMBINED PREDICTIONS
# --------------------------------------------------

combined_predictions = get_model_predictions(
    combined_model,
    X_test_combined,
    combined_thresholds
)

print("Combined predictions completed!")


# --------------------------------------------------
# VERIFY PREDICTION SHAPES
# --------------------------------------------------

print("\nPrediction shapes:")

print("Sound:", sound_predictions.shape)
print("Lyrics:", lyrics_predictions.shape)
print("Combined:", combined_predictions.shape)

print("\nAll final testing predictions completed!")

# ==================================================
# STEP 61: CALCULATE FINAL TEST SCORES
# ==================================================

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

print("\nCalculating final Words vs. Sound results...")


# --------------------------------------------------
# FUNCTION TO EVALUATE EACH MODEL
# --------------------------------------------------

def evaluate_model(model_name, predictions):

    micro_f1 = f1_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    precision = precision_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0
    )

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("Micro F1:", round(micro_f1, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))

    print("\nPerformance by activity:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=activities,
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Micro F1": micro_f1,
        "Macro F1": macro_f1,
        "Precision": precision,
        "Recall": recall
    }


# --------------------------------------------------
# EVALUATE ALL THREE MODELS
# --------------------------------------------------

sound_results = evaluate_model(
    "Sound-Only Model",
    sound_predictions
)

lyrics_results = evaluate_model(
    "Lyrics-Only Model",
    lyrics_predictions
)

combined_results = evaluate_model(
    "Combined Words + Sound Model",
    combined_predictions
)


# --------------------------------------------------
# CREATE FINAL COMPARISON TABLE
# --------------------------------------------------

final_results = pd.DataFrame([
    sound_results,
    lyrics_results,
    combined_results
])

print("\n" + "=" * 60)
print("FINAL WORDS VS. SOUND COMPARISON")
print("=" * 60)

print(
    final_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Identify the best model for each F1 metric
best_micro_model = final_results.loc[
    final_results["Micro F1"].idxmax(),
    "Model"
]

best_macro_model = final_results.loc[
    final_results["Macro F1"].idxmax(),
    "Model"
]

print("\nBest Micro F1 Model:", best_micro_model)
print("Best Macro F1 Model:", best_macro_model)

print("\nFinal model comparison completed!")

# ==================================================
# STEP 62: SAVE FINAL COMPARISON RESULTS
# ==================================================

print("\nSaving final comparison results...")

# Save overall model results
final_results.to_csv(
    "models/final_model_comparison.csv",
    index=False
)

print("Final comparison CSV saved successfully!")

# Save individual activity F1 scores
activity_results = pd.DataFrame({
    "Activity": activities,

    "Sound F1": f1_score(
        y_test,
        sound_predictions,
        average=None,
        zero_division=0
    ),

    "Lyrics F1": f1_score(
        y_test,
        lyrics_predictions,
        average=None,
        zero_division=0
    ),

    "Combined F1": f1_score(
        y_test,
        combined_predictions,
        average=None,
        zero_division=0
    )
})

activity_results.to_csv(
    "models/final_activity_comparison.csv",
    index=False
)

print("Activity comparison CSV saved successfully!")

print("\nFinal results saved!")

# ==================================================
# DEBUG: COMPARE ONE SONG WITH STREAMLIT
# ==================================================

print("\nChecking one song's predictions...")

# Change this number to the song index shown in Streamlit
song_index = 171140

# Find this song in the test dataset
matching_positions = np.where(
    test_indices == song_index
)[0]

if len(matching_positions) == 0:

    print("Song not found in test dataset.")

else:

    position = matching_positions[0]

    print("\nSong:", test_df.loc[song_index, "song"])
    print("Artist:", test_df.loc[song_index, "Artist(s)"])

    print("\nScaled sound features:")
    print(X_test_sound[position])

    # Run the same song through the saved sound model
    song_tensor = torch.tensor(
        X_test_sound[position:position + 1],
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = sound_model(song_tensor)

        probabilities = torch.sigmoid(outputs).numpy()[0]

    print("\nRaw model outputs:")
    print(outputs.numpy())

    print("\nProbabilities:")
    print(probabilities)

    print("\nPredictions:")
    print(probabilities >= sound_thresholds)

    print("\nActual dataset labels:")
    print(test_df.loc[song_index, activities])