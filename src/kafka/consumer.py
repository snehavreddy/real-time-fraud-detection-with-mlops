import json
import numpy as np
import time
import mlflow.sklearn
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from feast import FeatureStore
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import start_http_server, Counter, Gauge

# Prometheus Metrics
FRAUD_COUNT = Counter('fraud_detected_total', 'Total number of fraud transactions detected')
LEGIT_COUNT = Counter('legit_transactions_total', 'Total number of legitimate transactions')
PREDICTION_SCORE = Gauge('fraud_prediction_score', 'Latest fraud prediction score')
CONSUMER_LAG = Gauge('kafka_consumer_lag', 'Kafka consumer lag')
DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Data drift score from Evidently')

# Data collection for Evidently
collected_data = []
reference_data = None

# Load reference data from Parquet if available (Training Data)
parquet_path = "/feature_repo/data/transactions.parquet"
if os.path.exists(parquet_path):
    print(f"Loading reference data from {parquet_path} for drift monitoring...")
    try:
        reference_data = pd.read_parquet(parquet_path)
        # Filter for relevant columns to match incoming transaction format
        relevant_cols = ["amount", "transaction_time", "user_age", "is_international", "avg_30d_spending"]
        reference_data = reference_data[relevant_cols]
        print(f"Reference data loaded successfully with {len(reference_data)} rows.")
    except Exception as e:
        print(f"Failed to load reference data from parquet: {e}")
else:
    print(f"Reference parquet not found at {parquet_path}. Will fallback to first batch.")

# Start Prometheus metrics server on port 8001 early
print("Starting Prometheus metrics server on port 8001...")
start_http_server(8001)
print("Prometheus metrics server started.")

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
kafka_broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
feast_repo_path = os.getenv("FEAST_REPO_PATH", "/feature_repo")

mlflow.set_tracking_uri(mlflow_uri)
# Load the latest version of the model from the registry (with retry)
model_name = "fraud_model"
model_version = "latest"
model = None

print(f"Waiting for model '{model_name}' to be available in registry...")
while model is None:
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model not found yet, retrying in 10s... ({e})")
        time.sleep(10)

# Initialize Feast Feature Store
store = FeatureStore(repo_path=feast_repo_path)

def create_consumer():
    while True:
        try:
            consumer = KafkaConsumer(
                "transactions",
                bootstrap_servers=kafka_broker,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest", # start from beginning
                enable_auto_commit=True, # commit automatically
                group_id="fraud-group" # group of consumers
            )
            print("Consumer connected.")
            return consumer
        except NoBrokersAvailable:
            print("Waiting for Kafka...")
            time.sleep(1)

consumer = create_consumer()

for message in consumer:
    txn = message.value
    user_id = txn.get("user_id")
    
    # Fetch avg_30d_spending from Feast (Redis)
    print(f"Fetching features for user_id: {user_id}")
    try:
        feature_vector = store.get_online_features(
            features=["transaction_features:avg_30d_spending"],
            entity_rows=[{"user_id": user_id}]
        ).to_dict()
        print(f"Full feature vector for {user_id}: {feature_vector}")
        
        # Feast to_dict() returns a dict with lists as values
        avg_30d_spending = feature_vector.get("avg_30d_spending", [0.0])[0]
        # Sometimes Feast returns the feature name as is, or with the feature view name prefix
        if avg_30d_spending is None:
             avg_30d_spending = feature_vector.get("transaction_features:avg_30d_spending", [0.0])[0]
             
        if avg_30d_spending is None:
            avg_30d_spending = 0.0
            
    except Exception as e:
        print(f"Error fetching features from Feast: {e}")
        avg_30d_spending = 0.0

    print(f"Joined avg_30d_spending: {avg_30d_spending}")

    # Join features: 4 from Kafka + 1 from Feast
    features = np.array([[
        txn["amount"],
        txn["transaction_time"],
        txn["user_age"],
        txn["is_international"],
        avg_30d_spending
    ]])

    prediction = model.predict(features)[0]
    PREDICTION_SCORE.set(float(prediction))

    if prediction == 1:
        FRAUD_COUNT.inc()
        print(f"🚨 FRAUD DETECTED | User: {user_id} | Amount: {txn['amount']} | Avg Spend: {avg_30d_spending}")
    else:
        LEGIT_COUNT.inc()
        print(f"✅ Legit Transaction | User: {user_id} | Amount: {txn['amount']}")
    # Collect data for Evidently monitoring
    current_txn = {
        "amount": txn["amount"],
        "transaction_time": txn["transaction_time"],
        "user_age": txn["user_age"],
        "is_international": txn["is_international"],
        "avg_30d_spending": avg_30d_spending,
        "prediction": float(prediction)
    }
    collected_data.append(current_txn)

    # Every 50 transactions, check for drift
    if len(collected_data) >= 50:
        df_current = pd.DataFrame(collected_data)
        
        if reference_data is None:
            print("Fallback: Setting reference data from first batch...")
            relevant_cols = ["amount", "transaction_time", "user_age", "is_international", "avg_30d_spending"]
            reference_data = df_current[relevant_cols].copy()
        else:
            print("Calculating Data Drift report...")
            try:
                drift_report = Report(metrics=[DataDriftPreset()])
                
                # Filter current data to match reference features
                relevant_cols = ["amount", "transaction_time", "user_age", "is_international", "avg_30d_spending"]
                drift_report.run(reference_data=reference_data, current_data=df_current[relevant_cols])
                
                # Extract a simple score for Prometheus
                report_dict = drift_report.as_dict()
                drift_share = report_dict['metrics'][0]['result']['drift_share']
                DATA_DRIFT_SCORE.set(drift_share)
                print(f"📈 Data Drift share: {drift_share}")
            except Exception as e:
                print(f"Error calculating drift: {e}")
        
        collected_data = [] # Reset buffer