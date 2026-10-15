# facebook_post_prediciton

A machine learning project that applies **K-Means clustering** to Facebook Live post data to group posts with similar characteristics. The repository includes a Streamlit application, a Jupyter Notebook for analysis and model development, a dataset, and a saved clustering model.

## Overview

This project explores patterns in Facebook Live posts by clustering records based on selected numeric features. K-Means is an unsupervised learning algorithm, so it groups similar records without requiring a target label.

## Features

- Interactive Streamlit interface.
- Loads a saved K-Means clustering model.
- Uses Facebook Live post data for clustering and exploration.
- Includes a Jupyter Notebook for data preparation and clustering analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn (for analysis and visualization, if used in the notebook)
- Jupyter Notebook
- Pickle

## Dataset

The dataset file is:

```text
Live.csv
```

It contains Facebook Live post records and their associated engagement-related fields. Check the dataset columns and notebook for the exact fields selected for clustering and any preprocessing steps.

## Machine Learning Model

The project uses **K-Means clustering**, an unsupervised learning algorithm that partitions records into a specified number of clusters based on feature similarity.

The saved model file is:

```text
kmeans_model.pkl
```

The notebook `K_Means_Clustering_Live_Modified.ipynb` contains the analysis and clustering workflow. Consult the notebook for the chosen cluster count, feature selection, preprocessing, and any evaluation or visualization results. This README does not assume a particular cluster count or performance score.

## Repository Structure

```text
facebook_post_prediciton/
├── app.py
├── Live.csv
├── K_Means_Clustering_Live_Modified.ipynb
├── kmeans_model.pkl
└── README.md
```

The uploaded application file includes a timestamp in its filename. Rename it to `app.py` in the repository, or update the run command to match the filename you use.

## Installation

Install the main dependencies:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn jupyter
```

If the application or notebook uses additional libraries, install those as well.

## Run the Application

Place `app.py` and `kmeans_model.pkl` in the same project directory, then run:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit.

## How to Use

1. Start the Streamlit application.
2. Open the application in your browser.
3. Follow the available interface controls to explore or cluster Facebook Live post data.
4. Review the displayed clustering results and visualizations.

The precise controls and outputs depend on the implementation in `app.py`.

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and clustering logic |
| `Live.csv` | Facebook Live post dataset |
| `K_Means_Clustering_Live_Modified.ipynb` | Data analysis and K-Means workflow |
| `kmeans_model.pkl` | Saved K-Means model |
| `README.md` | Project documentation |

## Limitations

- Cluster quality depends on the selected features, scaling, and number of clusters.
- Clusters indicate similarity in the chosen data; they do not automatically establish causation or predict future engagement.
- Results depend on the dataset's coverage and quality.
- Only load Pickle files from trusted sources.

## Future Improvements

- Document the selected features and cluster count.
- Add clear descriptions of each cluster based on its engagement patterns.
- Compare different cluster counts using appropriate clustering metrics.
- Add validation and preprocessing checks for new data.
- Improve visualizations and deploy the application online.

## Conclusion

`facebook_post_prediciton` demonstrates an unsupervised machine learning workflow for exploring Facebook Live post data using K-Means clustering, with a saved model and a Streamlit interface.
