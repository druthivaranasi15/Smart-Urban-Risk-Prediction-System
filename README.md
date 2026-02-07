#  Smart Urban Risk Prediction System (Hybrid AI)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)

An advanced traffic safety system that predicts the **Severity Risk** of accidents based on live environmental conditions. This project shifts traffic management from **Reactive** (reporting crashes) to **Proactive** (preventing them).

##  Project Overview
Traffic accidents are dynamic; a safe road can become deadly within minutes due to rain or darkness. This system uses a **Hybrid Artificial Intelligence** model to analyze real-time data and warn authorities of high-risk zones *before* accidents occur.

### **The Hybrid Architecture**
We use an **Ensemble Approach** to ensure maximum accuracy:
1.  **Gradient Boosting (XGBoost):** Handles structured data (Road Type, Time) with 70% weight.
2.  **Deep Learning (TensorFlow/Keras):** Captures complex, non-linear patterns with 30% weight.
3.  **Calibration Layer:** A realism filter that adjusts predictions to match real-world accident probability.

##  Key Features
* **Real-Time Risk Analysis:** Instant prediction based on live inputs (Weather, Time, Lighting).
* **Proactive Alerts:** Visual "High Risk" warnings (Red Alert) for dangerous conditions.
* **Hybrid Voting System:** Combines two AI models for robust decision-making.
* **Interactive Dashboard:** Built with Streamlit for easy use by traffic police or city planners.

##  Screenshots

| Safe Scenario (Sunny/Noon) | High Risk Scenario (Rain/Night) |
|:--------------------------:|:-----------------------------:|
| *[Upload your Green Bar screenshot here]* | *[Upload your Red Bar screenshot here]* |

##  Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Smart-Urban-Risk-System.git](https://github.com/YOUR_USERNAME/Smart-Urban-Risk-System.git)
    cd Smart-Urban-Risk-System
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python -m streamlit run app.py
    ```

##  Tech Stack
* **Frontend:** Streamlit
* **Machine Learning:** XGBoost, Scikit-Learn
* **Deep Learning:** TensorFlow (Keras)
* **Data Processing:** Pandas, NumPy

## 🤝 Contributors
* **Druthi Varanasi** - Lead Developer

---
*Created for [Your College/Course Name] Final Project.* (preventing them).
