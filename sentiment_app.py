import streamlit as st
import pandas as pd
import joblib

# Load the saved model pipeline
model_filename = 'svm_model_pipeline.pkl'
pipeline = joblib.load(model_filename)

# Streamlit app
st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of user comments.")

# Text input for user comments
user_input = st.text_area("Enter your comment here:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input text (assuming you have a preprocess_text function)
        #cleaned_input = preprocess_text(user_input)

        # Make prediction
        prediction = pipeline.predict([user_input])[0]

        # Display the result
        st.write(f"Predicted Sentiment: {prediction}")
    else:
        st.write("Please enter a comment to analyze.")

# Upload CSV for batch predictions
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    if 'Comments' in new_df.columns:
        new_df['Comments'] = new_df['Comments'].astype(str)
        new_df['cleaned_comment'] = new_df['Comments'].apply(preprocess_text)

        # Make predictions
        new_df['predicted_sentiment'] = pipeline.predict(new_df['cleaned_comment'])

        # Display the predictions
        st.write(new_df[['Comments', 'predicted_sentiment']])
    else:
        st.write("The uploaded CSV file must contain a 'Comments' column.")
