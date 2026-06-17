# Intelligent IIoT Compressor Predictive Maintenance Platform

An end-to-end Machine Learning and Industrial Internet of Things (IIoT) analytics platform designed to monitor manufacturing compressor telemetry streams in real time. The platform processes continuous multi-variable sensor inputs to predict mechanical failure risks, classify complex fault signatures, and serve prescriptive maintenance playbooks to reduce costly operational downtime.

The core predictive engine features a **Random Forest Classifier** trained on **Amazon SageMaker** and served via a highly responsive **Streamlit Cloud** monitoring interface.

---

## 🚀 Key System Capabilities

* **Multi-Variable Risk Inference:** Moves beyond rigid, single-parameter alert limits by evaluating cross-feature relationships dynamically using machine learning.
* **Tiered SCADA-Style Alerting System:** Employs an industry-standard event layout utilizing non-intrusive `st.toast` notifications for minor telemetry variances (**ELEVATED**) and priority layout overrides (**CRITICAL**) for urgent system failures.
* **Cognitive Diagnostics & Breakdown:** Isolates complex industrial failure signatures including **Bearing Seizure**, **Piston Valve Leakage**, and **Intake Filtration Blockages** concurrently.
* **Prescriptive Maintenance Playbooks:** Generates actionable, step-by-step engineering guidelines dynamically calibrated to the real-time asset risk level.

---

## 🏗️ Architecture & Data Pipeline Blueprint

The platform bridges enterprise cloud machine learning infrastructure with front-end data applications:

1. **Telemetry Generation:** Simulates live continuous IIoT sensor array data (Vibration, Temperature, Suction/Discharge Pressures, Operating Hours).
2. **Model Training (AWS SageMaker):** An expanded data array of historical failure signatures is fed into a Random Forest Classifier architecture to build high-accuracy decision paths.
3. **Serialized Asset Deployment:** The optimized model brain is exported via `joblib` binaries (`compressor_risk_model.pkl`) and checked into the application workspace.
4. **Live Ingest & Inference Engine:** Streamlit reads the model asset, processes live dashboard slider arrays natively, evaluates safety heuristics overrides, and pushes interface state notifications instantly.

---

## 📂 Project Repository Structure

```text
compressor-health-monitoring/
│
├── .streamlit/
│   └── config.toml             # Streamlit workspace configuration parameters
│
├── models/
│   └── compressor_risk_model.pkl  # Trained production Random Forest model binary
│
├── app.py                      # Main production telemetry frontend dashboard application
├── compressor_clean_preprocessed.csv # Expanded industrial anomaly signature training dataset
├── requirements.txt            # Application container environment dependencies
└── README.md                   # Enterprise technical project documentation

