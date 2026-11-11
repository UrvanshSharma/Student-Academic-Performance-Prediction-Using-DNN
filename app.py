import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


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


try:
    # Load the models and preprocessing files created by the notebook.
    dnn_model = tf.keras.models.load_model("dnn_model.keras")
    random_forest = joblib.load("random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    metadata = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    st.error("Model files were not found. Run Student_Performance_Prediction.ipynb first.")
    st.stop()


st.title("🎓 Student Academic Performance Prediction")
st.write(
    "Enter a student's Mathematics-course details to estimate the final grade (out of 20) "
    "with a Deep Neural Network and a Random Forest model."
)
st.caption("The app loads pre-trained models only; it does not retrain them.")

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

    # Each model makes its own prediction from the same prepared row.
    dnn_grade = float(dnn_model.predict(prepared_features, verbose=0)[0][0])
    random_forest_grade = float(random_forest.predict(prepared_features)[0])

    # Grades in the original data range from 0 to 20.
    dnn_grade = float(np.clip(dnn_grade, 0, 20))
    random_forest_grade = float(np.clip(random_forest_grade, 0, 20))

    st.subheader("Prediction Results")
    dnn_column, forest_column = st.columns(2)
    with dnn_column:
        st.metric("Deep Neural Network", f"{dnn_grade:.2f} / 20")
        st.info(f"Category: {performance_category(dnn_grade)}")
    with forest_column:
        st.metric("Random Forest", f"{random_forest_grade:.2f} / 20")
        st.info(f"Category: {performance_category(random_forest_grade)}")
