import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import html

from scipy.sparse import csr_matrix, hstack


# app settings
torch.set_num_threads(1)

st.set_page_config(
    page_title="Words vs. Sound",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# activities and features
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


# pink theme
st.markdown("""
<style>

    .stApp {
        background-color: #FFF5F9;
        color: #503746;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.4rem;
        padding-bottom: 1rem;
    }

    h1, h2, h3 {
        color: #C34F85 !important;
        font-family: inherit !important;
        font-weight: 700 !important;
    }

    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.2rem !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #E46A9D;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #C34F85;
        color: white;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: #F1B6CE;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] {
        background-color: #FFEAF2;
        border: 1px solid #F3BDD2;
        border-radius: 12px;
    }

    .model-card {
        background-color: #FFFFFF;
        border: 1px solid #F3BDD2;
        border-top: 5px solid #E46A9D;
        border-radius: 14px;
        padding: 20px;
        min-height: 235px;
        box-shadow: 0 3px 12px rgba(195, 79, 133, 0.08);
    }

    .model-title {
        font-family: inherit;
        font-size: 1.35rem;
        color: #C34F85;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .model-description {
        font-size: 0.88rem;
        color: #94677C;
        margin-bottom: 18px;
    }

    .activity-tag {
        display: inline-block;
        background-color: #FFDCEB;
        color: #A83F70;
        padding: 6px 11px;
        border-radius: 8px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .song-banner {
        background-color: #FFE1ED;
        border-left: 4px solid #E46A9D;
        padding: 15px 20px;
        border-radius: 12px;
        margin: 15px 0;
    }

    .song-title {
        font-family: inherit;
        color: #B84478;
        font-size: 1.3rem;
        font-weight: 700;
    }

    .song-artist {
        color: #94677C;
        margin-top: 4px;
    }

    hr {
        border-color: #F3BDD2 !important;
    }

</style>
""", unsafe_allow_html=True)


# load songs from my test dataset
@st.cache_data
def load_songs():

    df = pd.read_csv("data/spotify_dataset.csv")

    # keep the same row order used for my saved test indices
    df = df.drop_duplicates().reset_index(drop=True)

    test_indices = np.load(
        "models/test_indices_grouped.npy"
    )

    test_songs = df.iloc[test_indices].copy()

    test_songs["display_name"] = (
        test_songs["song"].fillna("Unknown Song").astype(str)
        + " — "
        + test_songs["Artist(s)"].fillna("Unknown Artist").astype(str)
    )

    return test_songs


# sound model
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


# lyrics model
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


# combined model
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


# load saved models
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


# prepare song features
def prepare_features(song, scaler, vectorizer):

    # sound features
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

    # lyrics features
    lyrics = song["text"]

    if pd.isna(lyrics):
        lyrics = ""

    lyrics_features = vectorizer.transform(
        [str(lyrics)]
    )

    # combine sound and lyrics
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


# make predictions
def predict_model(model, features, thresholds):

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


# display model results in a card
def show_model_card(model_name, result):

    descriptions = {
        "Sound Only": "Using the song's audio characteristics",
        "Lyrics Only": "Using the words in the song",
        "Words + Sound": "Using both lyrics and sound features"
    }

    predictions = result["predictions"]
    probabilities = result["probabilities"]

    predicted_activities = [
        activity
        for activity, predicted in zip(
            ACTIVITIES,
            predictions
        )
        if predicted
    ]

    if predicted_activities:

        activity_html = "".join(
            f'<span class="activity-tag">{html.escape(activity)}</span>'
            for activity in predicted_activities
        )

    else:

        activity_html = (
            '<span style="color: #94677C;">'
            'No activities passed the prediction thresholds.'
            '</span>'
        )

    best_index = int(np.argmax(probabilities))

    best_activity = ACTIVITIES[best_index]
    best_score = probabilities[best_index]

    # keep HTML on one line so streamlit doesn't show it as code
    card_html = (
        '<div class="model-card">'
        f'<div class="model-title">{html.escape(model_name)}</div>'
        f'<div class="model-description">{html.escape(descriptions[model_name])}</div>'
        '<p><strong>Predicted activities</strong></p>'
        f'<div>{activity_html}</div>'
        '<p style="margin-top: 20px; font-size: 0.85rem;">'
        f'Highest score: {html.escape(best_activity)} ({best_score:.1%})'
        '</p>'
        '</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )


# app title
st.title("Words vs. Sound")

st.markdown(
    "**Does the music or the message matter more?**"
)

st.caption(
    "A machine learning project comparing how lyrics and sound "
    "features predict suitable listening activities."
)

st.divider()


# choose a song
songs = load_songs()

left, right = st.columns(
    [3, 1],
    vertical_alignment="bottom"
)

with left:

    selected_index = st.selectbox(
        "Pick a song",
        options=songs.index.tolist(),
        format_func=lambda index: songs.loc[
            index, "display_name"
        ],
        index=None,
        placeholder="Search by song title or artist..."
    )

with right:

    predict_clicked = st.button(
        "Compare Models",
        type="primary",
        use_container_width=True
    )

st.caption(
    f"Choose from {len(songs):,} songs "
    "in the held-out test dataset."
)


# run all three models
if predict_clicked:

    if selected_index is None:

        st.warning(
            "Please select a song first."
        )

    else:

        with st.spinner("Generating predictions..."):

            selected_song = songs.loc[selected_index]

            models, scaler, vectorizer = load_models()

            features = prepare_features(
                selected_song,
                scaler,
                vectorizer
            )

            actual_labels = selected_song[
                LABEL_COLUMNS
            ].to_numpy(dtype=int)

            all_results = {}

            for model_name in [
                "Sound Only",
                "Lyrics Only",
                "Words + Sound"
            ]:

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

            st.session_state["results"] = all_results

            st.session_state["song"] = {
                "title": str(selected_song["song"]),
                "artist": str(selected_song["Artist(s)"]),
                "actual_labels": actual_labels
            }


# display predictions
if "results" in st.session_state:

    results = st.session_state["results"]
    song_info = st.session_state["song"]

    song_title = html.escape(song_info["title"])
    song_artist = html.escape(song_info["artist"])

    # selected song banner
    banner_html = (
        '<div class="song-banner">'
        f'<div class="song-title">{song_title}</div>'
        f'<div class="song-artist">{song_artist}</div>'
        '</div>'
    )

    st.markdown(
        banner_html,
        unsafe_allow_html=True
    )

    st.subheader("What do the models think?")

    # three model cards side by side
    col1, col2, col3 = st.columns(
        3,
        gap="medium"
    )

    with col1:

        show_model_card(
            "Sound Only",
            results["Sound Only"]
        )

    with col2:

        show_model_card(
            "Lyrics Only",
            results["Lyrics Only"]
        )

    with col3:

        show_model_card(
            "Words + Sound",
            results["Words + Sound"]
        )

    st.subheader("Side-by-Side Comparison")

    comparison_df = pd.DataFrame({
        "Activity": ACTIVITIES,

        "Actual Label": [
            "Yes" if value else "No"
            for value in song_info["actual_labels"]
        ],

        "Sound": [
            "Yes" if value else "No"
            for value in results["Sound Only"]["predictions"]
        ],

        "Lyrics": [
            "Yes" if value else "No"
            for value in results["Lyrics Only"]["predictions"]
        ],

        "Combined": [
            "Yes" if value else "No"
            for value in results["Words + Sound"]["predictions"]
        ]
    })

    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Actual Label shows the original activity labels "
        "from the dataset."
    )

    # extra details without taking up space
    with st.expander("Detailed Model Scores"):

        scores_df = pd.DataFrame({
            "Activity": ACTIVITIES,
            "Sound": results["Sound Only"]["probabilities"],
            "Lyrics": results["Lyrics Only"]["probabilities"],
            "Combined": results["Words + Sound"]["probabilities"]
        })

        st.dataframe(
            scores_df.style.format({
                "Sound": "{:.1%}",
                "Lyrics": "{:.1%}",
                "Combined": "{:.1%}"
            }),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "These are the models' output scores. "
            "Each model uses its saved prediction thresholds "
            "to decide which activities to predict."
        )

    with st.expander("Overall Model Performance"):

        final_results = pd.read_csv(
            "models/final_model_comparison.csv"
        )

        st.dataframe(
            final_results,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "These scores were calculated using the full "
            "held-out testing dataset, not just this song."
        )


# footer
st.divider()

st.caption(
    "Words vs. Sound | CIS 377 | Cecily Davis"
)