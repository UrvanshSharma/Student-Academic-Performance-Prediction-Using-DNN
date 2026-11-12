# Student Academic Performance Prediction

A beginner-friendly machine learning project that predicts a student's final Mathematics grade (`G3`, out of 20) with a Deep Neural Network (DNN). It uses the UCI Student Performance Mathematics dataset.

## Project files

| File | Purpose |
| --- | --- |
| `student-mat.csv` | UCI Mathematics student data |
| `Student_Performance_Prediction.ipynb` | Data exploration, preprocessing, training, and evaluation |
| `app.py` | Streamlit form that trains a DNN at startup and makes predictions |
| `requirements.txt` | Python dependencies |

## Setup

Use Python 3.11, then open a terminal in this project folder and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The dataset is included. You can start the app right away:

```bash
streamlit run app.py
```

Streamlit will print the local address to open in your browser. The app trains the DNN from `student-mat.csv` when it starts, so the first page load takes a short time. It does not save a model file.

## Run the notebook

To inspect or retrain the project, launch Jupyter:

```bash
jupyter notebook
```

Open `Student_Performance_Prediction.ipynb` and run all cells from top to bottom to explore the dataset and train the DNN. The Streamlit app does not depend on notebook output files.

## Workflow

1. Check the dataset for missing values and duplicate rows.
2. Split the data into 80% training and 20% testing.
3. One-hot encode categorical attributes and standardize numerical attributes. The scaler is fit on training data only.
4. Compare DNN configurations with hidden layers of 32/16 and 64/32 neurons, using validation MAE and early stopping to select a configuration.
5. Evaluate the selected DNN on the held-out test split using MAE, MSE, RMSE, and R².

The Streamlit form collects the input attributes, applies the same encoding and scaling used during training, and shows the DNN prediction. Grade categories are Needs Improvement (0–9), Average (10–13), Good (14–16), and Excellent (17–20).

## Recorded evaluation

The latest notebook run selected the 32/16 DNN configuration. These results were calculated from predictions on its held-out test data (random state 42):

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| DNN (32/16) | 1.798 | 7.435 | 2.727 | 0.637 |

The results are included as a reference for this saved training run. Re-running training may produce slightly different scores depending on the software environment and training execution.

## Dataset source

UCI Machine Learning Repository, [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance). The file used here is `student-mat.csv`.
