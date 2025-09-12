from flask import Flask, request, jsonify, render_template
import requests
import time
import json
import os
import pickle
import csv
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load the pre-trained K-Means model and scaler
try:
    with open('kmeans_model.pkl', 'rb') as model_file:
        kmeans = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    model_is_loaded = True
    print("Machine learning model and scaler loaded successfully.")
except FileNotFoundError:
    print("Warning: Model files 'kmeans_model.pkl' or 'scaler.pkl' not found.")
    print("Please run 'train_model.py' first to generate these files.")
    model_is_loaded = False
except Exception as e:
    print(f"Error loading model files: {e}")
    model_is_loaded = False

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/ping_url', methods=['POST'])
def ping_url():
    """Pings a user-provided URL and returns the response time and status code."""
    url = request.json.get('url')
    if not url:
        return jsonify({"success": False, "message": "URL not provided"}), 400
    
    # Check if the URL has a scheme, if not, prepend 'https://'
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    try:
        start_time = time.time()
        # Ping the URL with a short timeout to prevent the server from hanging
        response = requests.get(url, timeout=10)
        end_time = time.time()
        
        response_time_ms = round((end_time - start_time) * 1000, 2)
        status_code = response.status_code

        return jsonify({
            "success": True,
            "response_time": response_time_ms,
            "status_code": status_code
        })

    except requests.exceptions.RequestException as e:
        # Handle connection errors (e.g., DNS failure, timeout)
        return jsonify({
            "success": False,
            "message": f"Ping failed: {str(e)}",
            "response_time": None,
            "status_code": None
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"An unknown error occurred: {str(e)}"}), 500

@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    """
    Receives collected data, performs K-Means clustering, and returns
    the results for plotting.
    """
    data = request.json.get('data')
    if not data:
        return jsonify({"success": False, "message": "No data received for analysis."}), 400

    if not model_is_loaded:
        return jsonify({"success": False, "message": "ML model not loaded. Please run 'train_model.py' first."}), 500

    # Filter out data points where ping failed
    valid_data = [item for item in data if item.get('response_time') is not None]
    
    if len(valid_data) < 3:
        return jsonify({"success": False, "message": "Not enough valid data points for clustering (min 3 required)."}), 400

    try:
        # Save collected data to a CSV file
        with open('analysis_data.csv', 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'response_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, item in enumerate(valid_data):
                writer.writerow({'timestamp': datetime.now().isoformat(), 'response_time': item['response_time']})
        
        # Extract the response times and reshape for the model
        response_times = [[item['response_time']] for item in valid_data]

        # Use the loaded scaler to transform the data
        scaled_data = scaler.transform(response_times)

        # Get the cluster labels for each data point
        clusters = kmeans.predict(scaled_data)

        # Get the cluster centroids
        cluster_centroids_scaled = kmeans.cluster_centers_
        cluster_centroids_original = scaler.inverse_transform(cluster_centroids_scaled).flatten().tolist()
        
        # Sort centroids to map them to labels
        sorted_centroids = sorted(cluster_centroids_original)
        centroid_map = {
            sorted_centroids[0]: 'Fast',
            sorted_centroids[1]: 'Medium',
            sorted_centroids[2]: 'Slow'
        }

        results = {
            "points": [],
            "centroids": []
        }
        
        # Add a random jitter for the y-axis to make the plot readable
        y_jitter = np.random.uniform(-0.5, 0.5, len(valid_data)).tolist()

        for i, item in enumerate(valid_data):
            # Get the original centroid value
            original_centroid_value = scaler.inverse_transform([cluster_centroids_scaled[clusters[i]]])[0][0]
            cluster_name = centroid_map.get(original_centroid_value, 'Unknown')
            
            results["points"].append({
                "x": item['response_time'],
                "y": y_jitter[i],
                "cluster": cluster_name
            })

        for centroid in sorted_centroids:
            results["centroids"].append({
                "x": centroid,
                "y": 0,
                "cluster": centroid_map.get(centroid)
            })
            
        return jsonify({
            "success": True, 
            "message": "Analysis complete.", 
            "results": results
        })

    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
