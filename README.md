# Industrial Predictive Maintenance System

An AI-based predictive maintenance solution for manufacturing and industrial equipment. This project utilizes machine learning to analyze multi-modal sensor telemetry (vibration, temperature, and current) to detect structural and electrical anomalies, predicting machinery failures before they occur.

## 🚀 Features
* **Synthetic Telemetry Generation:** Simulates realistic factory floor data stream linking interdependent physical variables (e.g., friction-induced thermal spikes).
* **Machine Learning Pipeline:** Employs a Scikit-learn Random Forest Classifier trained on non-linear operational failure boundaries.
* **Analytical Dashboard:** Complete Matplotlib-driven data visualization suite plotting sensor trends, feature importance weights, and confusion matrices.
* **Live Stream Simulation:** A real-time command-line stream monitor evaluating health thresholds with live operational risk probabilities and maintenance dispatch alerts.

---

## 🛠️ Tech Stack
* **Core Language:** Python 3
* **Data Engineering:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Matplotlib

---

## 📊 Feature Architecture
The system monitors three primary operational indicators to track machine degradation:

| Sensor Metric | Unit | Baseline | Risk Indicators Captured |
| :--- | :--- | :--- | :--- |
| **Vibration** | mm/s | 2.5 | Mechanical wear, misalignment, bearing degradation |
| **Temperature** | °C | 65.0 | Overheating, friction build-up, poor ventilation |
| **Current** | Amps | 12.0 | Electrical overload, rotor resistance, voltage sag |

---

## 📋 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR-USERNAME/industrial-predictive-maintenance.git](https://github.com/YOUR-USERNAME/industrial-predictive-maintenance.git)
cd industrial-predictive-maintenance
