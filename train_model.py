import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import sys

def train_and_save_model(data_file='analysis_data.csv'):
    """
    Reads data from a CSV file, trains a K-Means model, and saves
    the model and scaler to disk.
    """
    try:
        # Load the data from the CSV file
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        print("Please run the web application for at least a minute to generate the data.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{data_file}' is empty.")
        print("Please ensure the web application collected data correctly.")
        return

    # Drop rows with missing response times (ping failures)
    df.dropna(subset=['response_time'], inplace=True)
    
    if df.empty:
        print("Error: No valid data points found in the CSV file after cleaning.")
        return

    # Use the response_time column as the feature for clustering
    X = df[['response_time']].values

    # Scale the data using StandardScaler. This is important for K-Means.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the K-Means model
    print("Training K-Means model...")
    # n_init='auto' is used to suppress a scikit-learn warning in newer versions
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    print("Model training complete.")

    # Save the trained model and the scaler to disk
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    print("K-Means model saved to 'kmeans_model.pkl'")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to 'scaler.pkl'")

if __name__ == '__main__':
    train_and_save_model()
