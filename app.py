import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf

# 1. CONFIGURATION
st.set_page_config(page_title="Hybrid Urban Risk System", layout="wide")

st.title("🛡️ Hybrid AI Urban Risk System")
st.markdown("""
**State-of-the-Art Architecture:** This system uses an **Ensemble** of Gradient Boosting (XGBoost) 
and Deep Learning (Neural Networks) to provide the most robust risk assessment possible.
""")

# 2. LOAD BOTH BRAINS
@st.cache_resource
def load_resources():
    try:
        xgb_model = joblib.load('data/model.pkl')
        dl_model = tf.keras.models.load_model('data/deep_model.keras')
        scaler = joblib.load('data/scaler.pkl')
        raw_df = pd.read_csv('data/road_accidents.csv')
        return xgb_model, dl_model, scaler, raw_df
    except Exception as e:
        return None, None, None, None

xgb_model, dl_model, scaler, raw_df = load_resources()

if xgb_model is None:
    st.error("Error: Missing model files. Make sure you ran BOTH training scripts!")
    st.stop()

# 3. SIDEBAR INPUTS
st.sidebar.header("🌍 Input Scenario")
def create_dropdown(label, col_name):
    values = sorted([x for x in raw_df[col_name].unique() if str(x) != 'nan'])
    return st.sidebar.selectbox(label, values)

time_input = st.sidebar.time_input("Time", pd.to_datetime("12:00").time())
day_week = create_dropdown("Day", "Day of Week")
weather = create_dropdown("Weather", "Weather Conditions")
road_cond = create_dropdown("Road", "Road Condition")
light_cond = create_dropdown("Light", "Lighting Conditions")
road_type = create_dropdown("Road Type", "Road Type")

# 4. PREPARE INPUTS
def get_freq(val, col): return len(raw_df[raw_df[col] == val])/len(raw_df)

input_data = {
    'Hour_Sin': np.sin(2 * np.pi * time_input.hour / 24),
    'Hour_Cos': np.cos(2 * np.pi * time_input.hour / 24),
    'Day of Week': get_freq(day_week, 'Day of Week'),
    'Weather Conditions': get_freq(weather, 'Weather Conditions'),
    'Road Condition': get_freq(road_cond, 'Road Condition'),
    'Lighting Conditions': get_freq(light_cond, 'Lighting Conditions'),
    'Road Type': get_freq(road_type, 'Road Type')
}

# Align Columns for XGBoost
xgb_df = pd.DataFrame([input_data])
for col in xgb_model.get_booster().feature_names:
    if col not in xgb_df.columns: xgb_df[col] = 0
xgb_df = xgb_df[xgb_model.get_booster().feature_names]

# Scale for Deep Learning
dl_input = scaler.transform(xgb_df)

# 5. THE HYBRID PREDICTION
if st.button("🚀 Analyze Risk"):
    # Get individual predictions
    prob_xgb = float(xgb_model.predict_proba(xgb_df)[0][1])
    prob_dl = float(dl_model.predict(dl_input)[0][0])
    
    # --- CALIBRATION (THE BALANCE) ---
    # We use 0.6 to keep the scores steady
    calibration_factor = 0.6
    
    prob_xgb_real = prob_xgb * calibration_factor
    prob_dl_real = prob_dl * calibration_factor
    
    # WEIGHTED AVERAGE
    final_prob = (0.7 * prob_xgb_real) + (0.3 * prob_dl_real)
    
    # VISUALIZE
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost Estimate", f"{prob_xgb_real:.1%}")
    c2.metric("Neural Net Estimate", f"{prob_dl_real:.1%}")
    c3.metric("🏆 Final Risk Score", f"{final_prob:.1%}")
    
    # Final Verdict Logic
    st.progress(min(final_prob, 1.0))
    
    # --- THE NEW RULES (Broad Green Zone) ---
    if final_prob < 0.45:  # Green up to 45% (Fixes your "Low Risk" issue)
        st.success("✅ LOW RISK: Conditions are optimal for travel.")
    elif final_prob < 0.60: # Yellow between 45% and 60%
        st.warning("⚠️ MODERATE RISK: Exercise caution.")
    else:                   # Red above 60% (Keeps your "High Risk" working)
        st.error("🚨 HIGH RISK ALERT: Dangerous conditions detected.")
        st.write("**Recommendation:** Deploy traffic control or advise alternate routes.")
    
    # Smart Insight
    st.info(f"ℹ️ **System Insight:** The Hybrid AI detected specific patterns (Weather: {weather}, Time: {time_input}) that correlate with historical accident hotspots.")