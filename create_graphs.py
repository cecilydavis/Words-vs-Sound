import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# create a folder to save my graphs
os.makedirs("documents", exist_ok=True)

# graph 1: comparing the overall performance of my three models

# enter the final micro f1 scores from my test results
models = ["Sound", "Lyrics", "Combined"]
scores = [0.8344, 0.2555, 0.8457]

# create the first bar graph
plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores)

# add a title and labels
plt.title("Final Model Comparison")
plt.xlabel("Model")
plt.ylabel("Micro F1 Score")
plt.ylim(0, 1)

# display the exact score above each bar
for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.02,
        f"{score:.4f}",
        ha="center"
    )

# save the first graph without opening it
plt.tight_layout()
plt.savefig("documents/model_comparison.png", dpi=300)
plt.close()

print("Graph 1 saved successfully!")


# graph 2: comparing how my models performed on each activity

# load the final activity results from my csv file
df = pd.read_csv("models/final_activity_comparison.csv")

# shorten the activity names so they fit on the graph
activities = [
    "Party",
    "Work/Study",
    "Relaxation",
    "Exercise",
    "Driving"
]

# get the f1 scores for each model
sound_scores = df["Sound F1"]
lyrics_scores = df["Lyrics F1"]
combined_scores = df["Combined F1"]

# set up the positions for the bars
x = np.arange(len(activities))
width = 0.25

# create the second bar graph
plt.figure(figsize=(11, 6))

# add a bar for each model under every activity
plt.bar(x - width, sound_scores, width, label="Sound")
plt.bar(x, lyrics_scores, width, label="Lyrics")
plt.bar(x + width, combined_scores, width, label="Combined")

# add a title and labels
plt.title("Model Performance by Activity")
plt.xlabel("Activity")
plt.ylabel("F1 Score")

# label the activities and set the score range
plt.xticks(x, activities)
plt.ylim(0, 1.1)

# add a legend so I can tell the models apart
plt.legend()

# save the second graph in my documents folder
plt.tight_layout()
plt.savefig("documents/activity_comparison.png", dpi=300)

print("Graph 2 saved successfully!")

# display the second graph
plt.show()