import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from scipy.sparse import csr_matrix, hstack

# ==================================================
# SETTINGS
# ==================================================

torch.set_num_threads(1)

st.set_page_config(
    page_title="Words vs. Sound",
    page_icon="🎵",
    layout="wide"
)

ACTIVITIES = [
    "Party",
    "Work/Study",
    "Relaxation/Meditation",
    "Exercise",
    "Driving"
]

LABEL_COLUMNS = [
    "Good for Party",
    "Good for Work/Study",
    "Good for Relaxation/Meditation",
    "Good for Exercise",
    "Good for Driving"
]

SOUND_FEATURES = [
    "Tempo",
    "Loudness (db)",
    "Energy",
    "Danceability",
    "Positiveness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness"
]


# ==================================================
# LOAD SONGS
# ==================================================

@st.cache_data
def load_songs():

    df = pd.read_csv("data/spotify_dataset.csv")

    # Use the exact same cleaning as training
    df = df.drop_duplicates().reset_index(drop=True)

    test_indices = np.load(
        "models/test_indices_grouped.npy"
    )

    # Only use songs the models were not trained on
    test_songs = df.iloc[test_indices].copy()

    test_songs["display_name"] = (
        test_songs["song"].fillna("Unknown Song").astype(str)
        + " — "
        + test_songs["Artist(s)"].fillna("Unknown Artist").astype(str)
    )

    return test_songs


# ==================================================
# MODEL ARCHITECTURES
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


# ==================================================
# LOAD ALL THREE MODELS
# ==================================================

@st.cache_resource
def load_models():

    sound_model = SoundModel()
    lyrics_model = LyricsModel()
    combined_model = CombinedModel()

    sound_model.load_state_dict(
        torch.load(
            "models/sound_model_grouped.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    lyrics_model.load_state_dict(
        torch.load(
            "models/lyrics_model_5epochs.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    combined_model.load_state_dict(
        torch.load(
            "models/combined_model_grouped.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    sound_model.eval()
    lyrics_model.eval()
    combined_model.eval()

    scaler = joblib.load(
        "models/sound_scaler_grouped.pkl"
    )

    vectorizer = joblib.load(
        "models/lyrics_tfidf.pkl"
    )

    sound_thresholds = np.load(
        "models/sound_thresholds_grouped.npy"
    )

    lyrics_thresholds = np.load(
        "models/lyrics_thresholds_5epochs.npy"
    )

    combined_thresholds = np.load(
        "models/combined_thresholds_grouped.npy"
    )

    return {
        "Sound Only": (
            sound_model,
            sound_thresholds
        ),
        "Lyrics Only": (
            lyrics_model,
            lyrics_thresholds
        ),
        "Words + Sound": (
            combined_model,
            combined_thresholds
        )
    }, scaler, vectorizer


# ==================================================
# PREPARE SONG FEATURES
# ==================================================

def prepare_features(song, scaler, vectorizer):

    # Sound features
    sound_values = song[SOUND_FEATURES].copy()

    sound_values["Loudness (db)"] = str(
        sound_values["Loudness (db)"]
    ).replace("db", "").strip()

    sound_values = pd.to_numeric(
        sound_values,
        errors="coerce"
    )

    if sound_values.isna().any():
        raise ValueError(
            "This song has missing or invalid sound features."
        )

    sound_df = pd.DataFrame(
        [sound_values.to_numpy(dtype=float)],
        columns=SOUND_FEATURES
    )

    sound_scaled = scaler.transform(
        sound_df
    ).astype(np.float32)

    # Lyrics features
    lyrics = song["text"]

    if pd.isna(lyrics):
        lyrics = ""

    lyrics_features = vectorizer.transform(
        [str(lyrics)]
    )

    # Combined features: sound first, then lyrics
    combined_features = hstack(
        [
            csr_matrix(sound_scaled),
            lyrics_features
        ],
        format="csr"
    ).astype(np.float32)

    return {
        "Sound Only": sound_scaled,
        "Lyrics Only": lyrics_features,
        "Words + Sound": combined_features
    }


# ==================================================
# MAKE MODEL PREDICTIONS
# ==================================================

def predict_model(model, features, thresholds):

    # Convert sparse features to a dense array
    if hasattr(features, "toarray"):
        features = features.toarray()

    input_tensor = torch.from_numpy(
        np.asarray(features, dtype=np.float32)
    )

    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = torch.sigmoid(
            logits
        ).numpy()[0]

    predictions = probabilities >= thresholds

    return probabilities, predictions


# ==================================================
# WEBSITE HEADER
# ==================================================

st.title("🎵 Words vs. Sound")

st.subheader("Song Activity Prediction Dashboard")

st.write(
    "Can a song's lyrics, sound features, or both "
    "better predict which activities it's suited for?"
)

st.info(
    "Explore how three trained neural networks classify "
    "songs for different activities."
)

st.divider()


# ==================================================
# SONG SELECTION
# ==================================================

songs = load_songs()

st.header("1. Choose a Song")

selected_index = st.selectbox(
    "Search for a song by title or artist:",
    options=songs.index.tolist(),
    format_func=lambda index: songs.loc[
        index, "display_name"
    ],
    index=None,
    placeholder="Type a song title or artist..."
)

st.caption(
    f"Search through {len(songs):,} songs "
    "from the held-out testing dataset."
)

st.divider()


# ==================================================
# MODEL SELECTION
# ==================================================

st.header("2. Choose a Model")

model_choice = st.radio(
    "Select the model you want to explore:",
    [
        "Sound Only",
        "Lyrics Only",
        "Words + Sound",
        "Compare All Models"
    ],
    horizontal=True
)

st.divider()


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.header("3. Generate Predictions")

if st.button(
    "Predict Activities",
    type="primary"
):

    if selected_index is None:

        st.warning(
            "Please select a song before predicting."
        )

    else:

        selected_song = songs.loc[selected_index]

        st.success(
            "Predictions generated successfully!"
        )

        st.subheader(
            f"🎶 {selected_song['song']}"
        )

        st.write(
            "**Artist:**",
            selected_song["Artist(s)"]
        )

        # Load models and prepare features
        models, scaler, vectorizer = load_models()

        features = prepare_features(
            selected_song,
            scaler,
            vectorizer
        )

        # Get actual labels from the dataset
        actual_labels = selected_song[
            LABEL_COLUMNS
        ].to_numpy(dtype=int)

        # Determine which models to run
        if model_choice == "Compare All Models":

            selected_models = [
                "Sound Only",
                "Lyrics Only",
                "Words + Sound"
            ]

        else:

            selected_models = [model_choice]

        all_results = {}

        # Generate predictions
        for model_name in selected_models:

            model, thresholds = models[model_name]

            probabilities, predictions = predict_model(
                model,
                features[model_name],
                thresholds
            )

            all_results[model_name] = {
                "probabilities": probabilities,
                "predictions": predictions
            }

        # ==================================================
        # DISPLAY EACH MODEL'S RESULTS
        # ==================================================

        st.header("Prediction Results")

        for model_name, result in all_results.items():

            st.subheader(model_name)

            probabilities = result["probabilities"]
            predictions = result["predictions"]

            results_df = pd.DataFrame({
                "Activity": ACTIVITIES,
                "Prediction": [
                    "Yes" if value else "No"
                    for value in predictions
                ],
                "Model Score": [
                    f"{value:.1%}"
                    for value in probabilities
                ],
                "Dataset Label": [
                    "Yes" if value else "No"
                    for value in actual_labels
                ],
                "Correct?": [
                    "Yes" if prediction == actual else "No"
                    for prediction, actual in zip(
                        predictions,
                        actual_labels
                    )
                ]
            })

            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True
            )

            # ==================================================
            # BEST MATCH
            # ==================================================

            best_index = np.argmax(
                probabilities
            )

            best_activity = ACTIVITIES[
                best_index
            ]

            best_score = probabilities[
                best_index
            ]

            st.write(
                f"**Highest-scoring activity:** "
                f"{best_activity} ({best_score:.1%})"
            )

            if not predictions.any():

                st.caption(
                    "None of the five activities exceeded "
                    "this model's saved prediction thresholds."
                )

            # ==================================================
            # BAR CHART
            # ==================================================

            chart_df = pd.DataFrame({
                "Activity": ACTIVITIES,
                "Model Score": probabilities
            })

            st.bar_chart(
                chart_df.set_index("Activity")
            )

            st.divider()

        # ==================================================
        # COMPARE ALL THREE MODELS
        # ==================================================

        if model_choice == "Compare All Models":

            st.header("Model Comparison")

            comparison_df = pd.DataFrame({
                "Activity": ACTIVITIES,
                "Dataset Label": [
                    "Yes" if value else "No"
                    for value in actual_labels
                ]
            })

            for model_name, result in all_results.items():

                comparison_df[model_name] = [
                    "Yes" if value else "No"
                    for value in result["predictions"]
                ]

            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True
            )

            st.subheader(
                "Model Scores Compared"
            )

            scores_df = pd.DataFrame({
                model_name: result["probabilities"]
                for model_name, result in all_results.items()
            }, index=ACTIVITIES)

            st.bar_chart(scores_df)

        # ==================================================
        # SHOW FINAL TESTING RESULTS
        # ==================================================

        st.header("Overall Model Performance")

        final_results = pd.read_csv(
            "models/final_model_comparison.csv"
        )

        st.dataframe(
            final_results,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "These performance scores were calculated "
            "using the full held-out testing dataset, "
            "not just the selected song."
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Words vs. Sound | Machine Learning Project"
)