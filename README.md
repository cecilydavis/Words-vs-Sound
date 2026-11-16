# Words vs. Sound

## What My Project Does

For this project, I wanted to see if a song's lyrics, sound, or both would be better at predicting what activities the song is good for.

I used a Spotify dataset with over 500,000 songs and trained three different models.

The models predict whether a song is good for:
- Party
- Work/Study
- Relaxation/Meditation
- Exercise
- Driving

I made three models:
1. Sound Only - Uses things like tempo, energy, and danceability.
2. Lyrics Only - Uses the words in the song.
3. Words + Sound - Uses both the lyrics and sound features.

I also made an app where someone can search for a song and see what each model predicts.

## How to Install Everything

First, download this repository and open the folder in VS Code.

I put all the Python libraries I used in `requirements.txt`.

To install them, run:

```bash
python -m pip install -r requirements.txt
```

### Downloading the Dataset

I got my dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/devdope/900k-spotify/data?select=spotify_dataset.csv

The dataset is too big to upload to GitHub, so it has to be downloaded separately.

1. Download `spotify_dataset.csv` from Kaggle.
2. Make a folder called `data` inside the project folder.
3. Put the CSV file inside that folder.

It should look like this:

```text
Words-vs-Sound/
    data/
        spotify_dataset.csv
```

The dataset needs to be the same version I used because my saved test data uses specific rows from it.

## How I Trained My Models

I started by cleaning the dataset, checking for missing values, and removing duplicate rows.

After cleaning, I had 498,052 songs.

At first, I split the data randomly, but I found out that some of the same songs were showing up in both training and testing.

I fixed this by splitting the data based on the song title and artist so the same song wouldn't appear in multiple groups.

I split the data into about:
- 70% training
- 15% validation
- 15% testing

I used the training data to teach the models, the validation data to figure out which prediction thresholds worked best, and the testing data to see how well the models actually performed.

The saved splits are in the `models` folder.

### Training the Models

To train the models, run these files in order.

**1. Sound Model**

```bash
python main.py
```

This is where I started the project and trained my sound-only model.

**2. Lyrics Model**

```bash
python lyrics_model.py
```

This trains the model that only looks at lyrics. I used TF-IDF to turn the lyrics into numbers that the model could use.

**3. Combined Model**

```bash
python combined_model.py
```

This trains the model that uses both sound and lyrics.

I trained the final versions of all three models for five epochs.

## How to Test My Models

After training the models, run:

```bash
python final_comparison.py
```

This compares all three models using the testing data.

I looked at Micro F1, Macro F1, precision, and recall to see which model did the best.

### My Results

**Sound Only**
- Micro F1: 0.8344
- Macro F1: 0.8481
- Precision: 0.7322
- Recall: 0.9698

**Lyrics Only**
- Micro F1: 0.2555
- Macro F1: 0.2121
- Precision: 0.1739
- Recall: 0.4816

**Words + Sound**
- Micro F1: 0.8457
- Macro F1: 0.8361
- Precision: 0.7780
- Recall: 0.9264

The combined model had the best Micro F1 score, but the sound-only model had the best Macro F1 score.

The lyrics-only model didn't do nearly as well as the other two.

I saved the final results in:
- `models/final_model_comparison.csv`
- `models/final_activity_comparison.csv`

I also saved the trained models, so someone can use the app without having to train everything again.

## How to Run My App

I made a Streamlit app so someone can actually try out my models.

After installing the libraries and downloading the dataset, run:

```bash
python -m streamlit run app.py
```

This should open the app in a browser.

Once it's open:

1. Search for a song by its title or artist.
2. Pick which model you want to use.
3. Click **Predict Activities**.
4. Look at the predictions and scores.
5. You can also choose **Compare All Models** to see the results from all three.

The app only uses songs from my testing dataset.

## Documents Folder

I put the pictures, charts, and diagrams used in my report in the `documents` folder.

## My Code Files

- `main.py` - My sound-only model and the first steps of my project.
- `lyrics_model.py` - My lyrics-only model.
- `combined_model.py` - My model that uses both lyrics and sound.
- `final_comparison.py` - Tests and compares all three models.
- `app.py` - Runs my Streamlit app.
- `requirements.txt` - Lists the Python libraries I used.
- `models/` - Holds my saved models and results.
- `documents/` - Holds the pictures and diagrams for my report.