import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- MINIMALIST CONFIG ---
st.set_page_config(page_title="Minimal Predictor", layout="centered")

# Custom CSS for Light Minimalist Luxury
st.markdown("""
    <style>
    /* Light off-white background */
    .stApp { 
        background-color: #F8F9FA; 
        color: #2D2E2E; 
    }
    
    /* Elegant Clean Title */
    .clean-title { 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
        font-size: 32px; 
        font-weight: 300; 
        color: #1A1A1A; 
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }

    /* Soft Buttons */
    div.stButton > button:first-child { 
        background-color: #1A1A1A; 
        color: white; 
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 400;
        width: 100%;
    }

    /* Hide Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="clean-title">Predictive Intelligence</p>', unsafe_allow_html=True)

# --- 1. DATA UPLOAD ---
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Predict", df.columns)
    with col2:
        features = st.multiselect("Based on", [c for c in df.columns if c != target])

    if features:
        # Automatic numeric cleaning
        data = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(data) > 5:
            X = data[features]
            y = data[target]
            model = LinearRegression().fit(X, y)
            
            st.divider()
            
            # --- 2. INPUTS ---
            inputs = []
            for f in features:
                val = st.number_input(f"Value for {f}", value=float(data[f].mean()))
                inputs.append(val)
            
            if st.button("Calculate Prediction"):
                prediction = model.predict([inputs])[0]
                
                # Native Streamlit visual instead of HTML
                st.divider()
                st.metric(label=f"Estimated {target}", value=round(prediction, 2))
                st.balloons()
        else:
            st.info("Please select numerical columns with sufficient data to enable prediction.")
else:
    # Minimal landing state
    st.write("") 
    st.caption("Drop your CSV file here to begin.")
