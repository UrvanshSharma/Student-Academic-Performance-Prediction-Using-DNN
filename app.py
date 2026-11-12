from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")


def performance_category(grade):
    if grade <= 9:
        return "Needs Improvement"
    if grade <= 13:
        return "Average"
    if grade <= 16:
        return "Good"
    return "Excellent"


FIELD_LABELS = {
    "school": "School",
    "sex": "Student's sex",
    "age": "Student's age (years)",
    "address": "Home area",
    "famsize": "Family size",
    "Pstatus": "Parents' living arrangement",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "Mjob": "Mother's occupation",
    "Fjob": "Father's occupation",
    "reason": "Main reason for choosing the school",
    "guardian": "Main guardian",
    "traveltime": "Travel time to school",
    "studytime": "Weekly study time",
    "failures": "Past class failures",
    "schoolsup": "Extra school support",
    "famsup": "Family study support",
    "paid": "Paid extra math classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Plans for higher education",
    "internet": "Internet access at home",
    "romantic": "Currently in a relationship",
    "famrel": "Family relationship quality",
    "freetime": "Free time after school",
    "goout": "Time spent with friends",
    "Dalc": "Alcohol use on school days",
    "Walc": "Alcohol use on weekends",
    "health": "Current health",
    "absences": "Number of school absences",
    "G1": "First-period math grade (0–20)",
    "G2": "Second-period math grade (0–20)",
}

CATEGORY_LABELS = {
    "school": {"GP": "Gabriel Pereira school", "MS": "Mousinho da Silveira school"},
    "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban", "R": "Rural"},
    "famsize": {"LE3": "3 or fewer family members", "GT3": "More than 3 family members"},
    "Pstatus": {"T": "Parents live together", "A": "Parents live apart"},
    "Mjob": {
        "at_home": "At home", "health": "Healthcare", "other": "Other",
        "services": "Civil services", "teacher": "Teacher",
    },
    "Fjob": {
        "at_home": "At home", "health": "Healthcare", "other": "Other",
        "services": "Civil services", "teacher": "Teacher",
    },
    "reason": {
        "course": "Course preference", "home": "Close to home",
        "other": "Other", "reputation": "School reputation",
    },
    "guardian": {"father": "Father", "mother": "Mother", "other": "Other"},
}

NUMERIC_OPTION_LABELS = {
    "Medu": {0: "No formal education", 1: "Primary education", 2: "Grades 5–9", 3: "Secondary education", 4: "Higher education"},
    "Fedu": {0: "No formal education", 1: "Primary education", 2: "Grades 5–9", 3: "Secondary education", 4: "Higher education"},
    "traveltime": {1: "Under 15 minutes", 2: "15–30 minutes", 3: "30–60 minutes", 4: "Over 60 minutes"},
    "studytime": {1: "Under 2 hours/week", 2: "2–5 hours/week", 3: "5–10 hours/week", 4: "Over 10 hours/week"},
    "failures": {0: "None", 1: "One", 2: "Two", 3: "Three or more"},
    "famrel": {1: "Very bad", 2: "Bad", 3: "Average", 4: "Good", 5: "Excellent"},
    "freetime": {1: "Very low", 2: "Low", 3: "Average", 4: "High", 5: "Very high"},
    "goout": {1: "Very rarely", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Very often"},
    "Dalc": {1: "Very low", 2: "Low", 3: "Average", 4: "High", 5: "Very high"},
    "Walc": {1: "Very low", 2: "Low", 3: "Average", 4: "High", 5: "Very high"},
    "health": {1: "Very bad", 2: "Bad", 3: "Fair", 4: "Good", 5: "Very good"},
}

def render_student_input(column, default_value, metadata):
    """Render the dataset code with a short explanation beside each option."""
    label = FIELD_LABELS.get(column, column.replace("_", " ").title())
    if column in metadata["categorical_columns"]:
        options = metadata["categorical_options"][column]
        default_index = options.index(default_value) if default_value in options else 0
        return st.selectbox(
            label,
            options,
            index=default_index,
            format_func=lambda value: (
                f"{value} ({CATEGORY_LABELS[column][value]})"
                if value in CATEGORY_LABELS.get(column, {})
                else value.replace("_", " ").title()
            ),
            key=column,
        )

    if column in NUMERIC_OPTION_LABELS:
        option_labels = NUMERIC_OPTION_LABELS[column]
        options = list(option_labels.keys())
        default_number = int(round(float(default_value)))
        default_index = options.index(default_number) if default_number in options else 0
        return st.selectbox(
            label,
            options,
            index=default_index,
            format_func=lambda value: f"{value} ({option_labels[value]})",
            key=column,
        )

    default_number = int(round(float(default_value)))
    return st.number_input(
        label,
        value=default_number,
        step=1,
        key=column,
    )


@st.cache_resource(show_spinner="Training the Deep Neural Network…")
def train_model(data_path, data_modified_time):
    """Train the DNN and prepare the form data once for each app startup."""
    del data_modified_time  # Include the CSV's timestamp in Streamlit's cache key.
    tf.keras.utils.set_random_seed(42)

    data = pd.read_csv(data_path, sep=";")
    features_raw = data.drop("G3", axis=1)
    grades = data["G3"]
    numeric_columns = features_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features_raw.select_dtypes(exclude=np.number).columns.tolist()

    features = pd.get_dummies(features_raw, columns=categorical_columns, dtype=float)
    scaler = StandardScaler()
    features[numeric_columns] = scaler.fit_transform(features[numeric_columns])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        features, grades, validation_split=0.20, epochs=40, batch_size=16,
        callbacks=[early_stop], verbose=0,
    )

    metadata = {
        "model_columns": features.columns.tolist(),
        "raw_columns": features_raw.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_options": {
            column: sorted(features_raw[column].unique())
            for column in categorical_columns
        },
        "defaults": {
            column: (
                features_raw[column].median()
                if column in numeric_columns
                else features_raw[column].mode()[0]
            )
            for column in features_raw.columns
        },
    }
    return model, scaler, metadata


try:
    DATA_PATH = "student-mat.csv"
    dnn_model, scaler, metadata = train_model(
        DATA_PATH, Path(DATA_PATH).stat().st_mtime_ns
    )
except FileNotFoundError:
    st.error("student-mat.csv was not found. Keep it in the same folder as app.py.")
    st.stop()


st.title("🎓 Student Academic Performance Prediction")
st.write(
    "Enter a student's Mathematics-course details to estimate the final grade (out of 20) "
    "with a Deep Neural Network."
)
st.caption("The app trains the DNN when it starts; it does not save model files.")

with st.form("student_details_form"):
    # Start with the saved typical values, then let the user change them.
    student_details = {}
    left_column, right_column = st.columns(2)

    for index, column in enumerate(metadata["raw_columns"]):
        destination = left_column if index % 2 == 0 else right_column
        with destination:
            student_details[column] = render_student_input(
                column, metadata["defaults"][column], metadata
            )

    submitted = st.form_submit_button("Predict Final Grade", use_container_width=True)

if submitted:
    # Put the form values into one table row.
    student_row = pd.DataFrame([student_details])

    # Turn text categories into 0/1 columns, like during notebook training.
    prepared_features = pd.get_dummies(
        student_row, columns=metadata["categorical_columns"], dtype=float
    )

    # Put the columns in the saved training order.
    prepared_features = prepared_features.reindex(
        columns=metadata["model_columns"], fill_value=0
    ).astype(float)

    # Scale number columns using the scaler saved by the notebook.
    prepared_features[metadata["numeric_columns"]] = scaler.transform(
        prepared_features[metadata["numeric_columns"]]
    )

    # The DNN predicts the final grade from the prepared row.
    dnn_grade = float(dnn_model.predict(prepared_features, verbose=0)[0][0])

    # Grades in the original data range from 0 to 20.
    dnn_grade = float(np.clip(dnn_grade, 0, 20))

    st.subheader("Prediction Results")
    st.metric("Deep Neural Network", f"{dnn_grade:.2f} / 20")
    st.info(f"Category: {performance_category(dnn_grade)}")
