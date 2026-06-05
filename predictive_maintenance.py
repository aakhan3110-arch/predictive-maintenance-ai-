import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import time

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. SIMULATED SENSOR DATA GENERATION
# ==========================================
def generate_sensor_data(num_samples=1000):
    """
    Simulates industrial sensor data for vibration, temperature, and current.
    Introduces synthetic failures based on physical anomalies.
    """
    print("[*] Generating simulated sensor data...")
    
    # Generate a timeline
    timestamps = pd.date_range(start="2026-01-01", periods=num_samples, freq="min")
    
    # Base normal operating conditions
    vibration = np.random.normal(loc=2.5, scale=0.5, size=num_samples)  # mm/s
    temperature = np.random.normal(loc=65.0, scale=5.0, size=num_samples)  # °C
    current = np.random.normal(loc=12.0, scale=1.5, size=num_samples)  # Amps
    
    # Inject synthetic anomalies/failures (e.g., bearing wear out, overheating)
    # Failure condition 1: High vibration and high temperature (Bearing wear)
    # Failure condition 2: Spiking current and temperature (Electrical overload)
    failures = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Inject deliberate anomalies in 8% of the data
        if i % 12 == 0:
            vibration[i] += np.random.uniform(2.0, 4.0)
            temperature[i] += np.random.uniform(15.0, 25.0)
            if vibration[i] > 4.5 and temperature[i] > 78.0:
                failures[i] = 1
                
        elif i % 19 == 0:
            current[i] += np.random.uniform(5.0, 10.0)
            temperature[i] += np.random.uniform(10.0, 20.0)
            if current[i] > 18.0 or temperature[i] > 82.0:
                failures[i] = 1

    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Vibration_mm_s': vibration,
        'Temperature_C': temperature,
        'Current_A': current,
        'Failure': failures
    })
    
    print(f"[+] Data generation complete. Total samples: {len(df)} | Failures injected: {df['Failure'].sum()}")
    return df

# ==========================================
# 2. MODEL TRAINING & EVALUATION
# ==========================================
def train_predictive_model(df):
    """
    Trains a Random Forest Classifier to predict machine failure based on sensor inputs.
    """
    print("\n[*] Training Random Forest Classifier...")
    
    # Features and Target
    X = df[['Vibration_mm_s', 'Temperature_C', 'Current_A']]
    y = df['Failure']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    
    print("\n=== Model Evaluation Metrics ===")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

# ==========================================
# 3. FAILURE PREDICTION DASHBOARD
# ==========================================
def plot_dashboard(df, model, X_test, y_test, y_pred):
    """
    Generates a performance and analytical dashboard using Matplotlib.
    """
    print("\n[*] Compiling Maintenance Dashboard visual assets...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Industrial Predictive Maintenance Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Sensor Run Charts (First 150 points for visual clarity)
    sample_df = df.iloc[:150]
    axes[0, 0].plot(sample_df['Timestamp'], sample_df['Temperature_C'], label='Temp (°C)', color='coral')
    axes[0, 0].plot(sample_df['Timestamp'], sample_df['Vibration_mm_s'] * 10, label='Vibration (mm/s x10)', color='purple')
    # Mark real failures
    fail_points = sample_df[sample_df['Failure'] == 1]
    axes[0, 0].scatter(fail_points['Timestamp'], fail_points['Temperature_C'], color='red', marker='X', s=100, label='Actual Failure')
    axes[0, 0].set_title('Sensor Telemetry & Failure Events Over Time')
    axes[0, 0].set_xlabel('Timestamp')
    axes[0, 0].set_ylabel('Sensor Metrics')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # Plot 2: Feature Importance
    importances = model.feature_importances_
    features = X_test.columns
    axes[0, 1].barh(features, importances, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 1].set_title('Feature Importance (Random Forest)')
    axes[0, 1].set_xlabel('Relative Importance Weight')
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Failure'])
    cm_display.plot(ax=axes[1, 0], cmap='Blues', values_format='d')
    axes[1, 0].set_title('Confusion Matrix')
    
    # Plot 4: Data Distribution (Normal vs Failure states)
    axes[1, 1].boxplot([df[df['Failure'] == 0]['Temperature_C'], df[df['Failure'] == 1]['Temperature_C']], labels=['Normal', 'Failure'])
    axes[1, 1].set_title('Temperature Distribution Across States')
    axes[1, 1].set_ylabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. LIVE MAINTENANCE ALERTS SYSTEM SIMULATION
# ==========================================
def simulate_live_stream(model):
    """
    Simulates a live incoming data stream from factory floor machinery, 
    evaluating health thresholds in real-time.
    """
    print("\n" + "="*50)
    print("STARTING LIVE FACTORY TELEMETRY STREAM SIMULATION")
    print("="*50)
    
    # Test cases: [Vibration, Temperature, Current]
    simulated_stream = [
        [2.1, 62.3, 11.8],  # Normal operational telemetry
        [2.8, 68.1, 13.2],  # Normal operational telemetry
        [5.8, 84.2, 12.1],  # CRITICAL: High vibration & temperature spike
        [2.3, 64.0, 11.5],  # Recovered/Normal operational telemetry
        [2.2, 83.5, 22.4]   # CRITICAL: Severe Overcurrent & Overheating
    ]
    
    for i, sample in enumerate(simulated_stream, 1):
        time.sleep(0.8)  # Mimic real-world polling interval
        
        # Format input for prediction
        input_data = pd.DataFrame([sample], columns=['Vibration_mm_s', 'Temperature_C', 'Current_A'])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        print(f"\n[Reading #{i}] Vib: {sample[0]}mm/s | Temp: {sample[1]}°C | Curr: {sample[2]}A")
        
        if prediction == 1:
            print(f" >>> [ALERT] CRITICAL MACHINERY ANOMALY DETECTED! <<<")
            print(f"     Failure Probability: {probability:.2%}")
            print(f"     Action Required: Schedule immediate predictive maintenance dispatch.")
        else:
            print(f" Status: Operational Nominal (Failure Prob: {probability:.2%})")

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Step 1: Generate Data
    sensor_df = generate_sensor_data(num_samples=1200)
    
    # Step 2: Model Engineering
    rf_model, X_test, y_test, y_pred = train_predictive_model(sensor_df)
    
    # Step 3: Analytics Visualization
    plot_dashboard(sensor_df, rf_model, X_test, y_test, y_pred)
    
    # Step 4: Live Streaming Verification Test
    simulate_live_stream(rf_model)
