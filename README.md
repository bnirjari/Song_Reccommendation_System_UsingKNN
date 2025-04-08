# Song Recommendation System using KNN

A content-based music recommendation engine that suggests songs based on user preferences and audio feature similarity using K-Nearest Neighbors (KNN) and clustering techniques.
---

## Overview

This project implements a **K-Nearest Neighbors (KNN)** based recommendation engine that identifies songs similar to a given input track. It leverages the **cosine similarity** metric to capture feature closeness and suggests top similar songs. The system also includes clustering techniques like **KMeans** and **Hierarchical Clustering** for performance comparison.

---

##  Dataset

- **Source**: [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Features Used**:
  - `danceability`, `energy`, `acousticness`, `valence`, `tempo`, `liveness`, `speechiness`, `instrumentalness`, `loudness`, etc.
- **Target**: `track_genre` (for clustering comparison, not used in recommendation)
- ~114,000 songs with metadata and numeric audio features.
- **Preprocessing steps**:
  - Removed irrelevant columns (Unnamed: 0, track_id, etc.)
  - Normalized features using MinMaxScaler
  - Encoded categorical columns and handled missing values.
---

## ⚙ Tech Stack

- Python (Pandas, NumPy, Sklearn, Seaborn, Matplotlib)
- Jupyter Notebook
- Gradio (for web interface)

---

## Model Used

- **KNN (K-Nearest Neighbors)** with **Cosine Similarity**
  - Finds songs with the closest feature space to the input song
  - `k` optimized using the Elbow Method

- **KMeans Clustering**
  - Groups songs into clusters based on audio feature similarity
  - Identifies the general vibe or genre of a song
  - Optimal number of clusters (k) chosen using the Elbow Method

---

## Evaluation Metrics

Since this is not a classification task, we use clustering and similarity-based metrics:
- **Silhouette Score** (for KMeans and KNN)
- **Visualizations**: Correlation Heatmap, Elbow Curve, Silhouette Comparison
- **Qualitative**: Relevance, consistency, and diversity of recommendations

---

##  Features

-  Search by song title and receive top N similar songs
-  Search by mood/keyword using substring matching
-  Compare clustering quality using silhouette scores
-  Gradio web interface for interactive recommendations
-  Model persistence with `pickle`

---
## Visualizations

-  Histograms for all numerical features.
-  Correlation heatmap to visualize feature interrelationships.
-  Silhouette score graph to evaluate clustering.
-  Elbow curve for optimal K selection.

---
## Project Structure

song-recommendation-knn/
│
├── requirements.txt                             # Python dependencies
│
├── Data/
│   └── dataset.csv                              # Song dataset used for modeling
│
├── Notebook/
│   └── project_final.ipynb                      # Main notebook with full implementation
│
├── docs/
│   └── Song_Recommendation_System_Documentation.pdf   # Project report/documentation
│
├── src/
│   └── Data_Preprocessing.ipynb                 # Notebook for data cleaning & preparation
│
├── Models/
│   ├── kmeans_model.pkl                         # Saved K-Means model
│   └── nearest_neighbors_model.pkl              # Saved KNN model

---
## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/song-recommendation-knn.git
   cd song-recommendation-knn
   
2. Install dependencies:
  - pip install -r requirements.txt

3. Launch Jupyter Notebook and run project_final.ipynb


## Author
- Nirjari Bhatt
