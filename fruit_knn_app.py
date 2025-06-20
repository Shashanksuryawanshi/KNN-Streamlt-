import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Sample Data
data = {
    "weight": [150, 170, 140, 120, 130, 110, 180, 160, 100],
    "color_score": [0.85, 0.87, 0.9, 0.6, 0.55, 0.65, 0.45, 0.4, 0.35],
    "label": ["apple", "apple", "apple", "banana", "banana", "banana", "orange", "orange", "orange"]
}
df = pd.DataFrame(data)

# Features & Labels
X = df[["weight", "color_score"]]
y = df["label"]

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Streamlit UI
st.title("🍎🍌🍊 Fruit Classifier using KNN")
st.write("Predict whether a fruit is Apple, Banana or Orange based on its weight and color score.")

# User Inputs
weight = st.slider("Fruit Weight (grams)", 80, 200, 130)
color_score = st.slider("Color Score (0 to 1)", 0.3, 1.0, 0.6)

# Prediction
if st.button("Predict Fruit"):
    prediction = model.predict([[weight, color_score]])[0]
    st.success(f"The model predicts this fruit is a **{prediction.upper()}**")

# Show Data (Optional)
if st.checkbox("Show training data"):
    st.dataframe(df)
