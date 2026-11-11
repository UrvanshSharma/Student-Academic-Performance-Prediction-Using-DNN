# Student Academic Performance Prediction

A beginner-friendly machine learning project that predicts a student's final Mathematics grade (`G3`, out of 20) with a Deep Neural Network (DNN) and compares it with a Random Forest regressor. It uses the UCI Student Performance Mathematics dataset.

## Project files

| File | Purpose |
| --- | --- |
| `student-mat.csv` | UCI Mathematics student data |
| `Student_Performance_Prediction.ipynb` | Data exploration, preprocessing, training, evaluation, and artifact saving |
| `app.py` | Streamlit form for getting predictions from both saved models |
| `dnn_model.keras` | Trained DNN model |
| `random_forest.pkl` | Trained Random Forest model |
| `scaler.pkl` | Fitted scaler for numerical features |
| `feature_columns.pkl` | Feature names, order, input options, and defaults used by the app |
| `requirements.txt` | Python dependencies |

## Setup

Use Python 3.11, then open a terminal in this project folder and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The dataset and trained model files are included. You can start the app right away:

```bash
streamlit run app.py
```

Streamlit will print the local address to open in your browser. The app loads the trained models and preprocessing files; it does not retrain them on startup.

## Run or retrain the notebook

To inspect or retrain the project, launch Jupyter:

```bash
jupyter notebook
```

Open `Student_Performance_Prediction.ipynb` and run all cells from top to bottom. The notebook reads `student-mat.csv` from this folder and saves or replaces `dnn_model.keras`, `random_forest.pkl`, `scaler.pkl`, and `feature_columns.pkl` in this same folder. Restart the Streamlit app after retraining so it loads the newly saved artifacts.

## Workflow

1. Check the dataset for missing values and duplicate rows.
2. Split the data into 80% training and 20% testing.
3. One-hot encode categorical attributes and standardize numerical attributes. The scaler is fit on training data only.
4. Compare DNN configurations with hidden layers of 32/16 and 64/32 neurons, using validation MAE and early stopping to select a configuration.
5. Train a Random Forest on the same training split.
6. Compare both models on the held-out test split using MAE, MSE, RMSE, and R².

The Streamlit form collects the input attributes, applies the saved encoding, feature order, and scaling, and shows predictions from both models. Grade categories are Needs Improvement (0–9), Average (10–13), Good (14–16), and Excellent (17–20).

## Recorded evaluation

The latest notebook run selected the 32/16 DNN configuration. These results were calculated from predictions on its held-out test data (random state 42):

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| DNN (32/16) | 1.798 | 7.435 | 2.727 | 0.637 |
| Random Forest | 1.214 | 4.035 | 2.009 | 0.803 |

The results are included as a reference for this saved training run. Re-running training may produce slightly different scores depending on the software environment and training execution.

## Dataset source

UCI Machine Learning Repository, [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance). The file used here is `student-mat.csv`.
