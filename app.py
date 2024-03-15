import streamlit as st
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder

# Load the data
def load_data():
    file_path = r"C:\Users\Darshan\Downloads\turnover.csv"
    try:
        # Try reading the file with UTF-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try reading with ISO-8859-1 encoding (Latin-1)
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        except Exception as e:
            # Handle other exceptions
            st.error(f"An error occurred: {str(e)}")
            return None
    return df

# Function to create Kaplan-Meier survival curve
def create_km_survival_curve(data, duration_col, event_col):
    kmf = KaplanMeierFitter()
    kmf.fit(data[duration_col], event_observed=data[event_col])
    plt.figure(figsize=(10, 6))
    kmf.plot()
    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    st.pyplot(plt.gcf())

# Function to create Kaplan-Meier survival curve profession-wise
def create_profession_wise_km_survival_curve(data, profession):
    # Filter DataFrame for the selected profession
    profession_df = data[data['profession'] == profession]
    
    # Initialize KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(profession_df['stag'], event_observed=None)  # No events observed
    
    # Plot Kaplan-Meier survival curve for the selected profession
    plt.figure(figsize=(10, 6))
    kmf.plot()
    plt.title(f'Kaplan-Meier Survival Curve for {profession}')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    st.pyplot(plt.gcf())

# Function to show profession-wise survival probabilities
def show_profession_wise_probabilities(data, profession):
    # Filter DataFrame for the selected profession
    profession_df = data[data['profession'] == profession]
    
    # Initialize KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(profession_df['stag'], event_observed=None)  # No events observed

    # Calculate the survival probability for each record in the selected profession
    survival_probabilities = kmf.survival_function_at_times(profession_df['stag']).values.flatten()
    
    # Print survival probabilities for the selected profession
    st.write(f"Survival probabilities for {profession}:")
    for i, prob in enumerate(survival_probabilities):
        st.write(f"Month {i + 1}: {prob}")

# Main function for Streamlit app
def main():
    st.title('Predictive Modeling of Employee Retention Using Survival Analysis')

    # Load the data
    df = load_data()

    if df is not None:
        # Display some basic info about the data
        st.write("Data Info:")
        st.write(df.head())
        st.write("Data Shape:", df.shape)

        # Add option to display the Kaplan-Meier survival curve
        st.subheader("Kaplan-Meier Survival Curve")
        st.write("Select the columns for duration and event:")
        duration_col = st.selectbox("Select Duration Column", df.columns)
        event_col = st.selectbox("Select Event Column", df.columns)
        if st.button("Generate Kaplan-Meier Survival Curve"):
            create_km_survival_curve(df, duration_col, event_col)
        
        # Add option to display profession-wise Kaplan-Meier survival curve
        st.subheader("Profession-wise Kaplan-Meier Survival Curve")
        profession_km_dropdown = st.selectbox("Select Profession", df['profession'].unique(), key='km_dropdown')
        if st.button("Generate Profession-wise Kaplan-Meier Survival Curve"):
            create_profession_wise_km_survival_curve(df, profession_km_dropdown)
            
        # Add option to show profession-wise survival probabilities
        st.subheader("Profession-wise Survival Probabilities")
        profession_prob_dropdown = st.selectbox("Select Profession", df['profession'].unique(), key='prob_dropdown')
        if st.button("Show Profession-wise Survival Probabilities"):
            show_profession_wise_probabilities(df, profession_prob_dropdown)

# Run the app
if __name__ == '__main__':
    main()
