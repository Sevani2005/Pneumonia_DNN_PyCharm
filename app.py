import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS with Minimalist Black & White Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }

    /* Container styling - Clean white with subtle shadow */
    .glass-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid #e5e5e5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Title styling - Bold Black */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #111111;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Button styling - Black */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #000000;
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        border: 1px solid #000000;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:hover {
        background-color: #333333;
        border-color: #333333;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1.5rem;
        animation: scaleIn 0.4s ease-out;
        border-left: 5px solid;
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.98); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Minimalist result styles */
    .normal {
        background-color: #f7fff9;
        color: #1a1a1a;
        border-color: #000000;
        border: 1px solid #e5e5e5;
        border-left: 5px solid #000000;
    }
    
    .pneumonia {
        background-color: #fff5f5;
        color: #1a1a1a;
        border: 1px solid #e5e5e5;
        border-left: 5px solid #000000;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .confidence-text {
        font-size: 1rem;
        color: #666666;
        font-weight: 500;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #f9fafb;
        border-radius: 8px;
        border: 1px dashed #d1d5db;
    }
    
    /* Stats cards - Minimal */
    .stat-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e5e5;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        border-color: #000000;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        color: #666666;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Image container */
    .image-container {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e5e5e5;
        margin: 1rem 0;
        background-color: #f9fafb;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f9fafb;
        border-right: 1px solid #e5e5e5;
    }
    
    /* Alerts */
    .stAlert {
        background-color: #f9fafb;
        border: 1px solid #e5e5e5;
        color: #000000;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">PNEUMONIA DETECTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Radiological Assistant</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### SYSTEM INFO")
    st.markdown("""
    - **Model:** ConvNet (CNN)
    - **Version:** 2.0.0
    - **Status:** Active
    """)
    
    st.markdown("---")
    st.markdown("### INSTRUCTIONS")
    st.markdown("""
    1. Upload Chest X-Ray
    2. Review Pre-processed Image
    3. Run Analysis
    """)
    


# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    # Load the model
    MODEL_PATH = "pneumonia_model.keras"
    
    @st.cache_resource
    def load_my_model():
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)
        return None
    
    model = load_my_model()
    
    if model is None:
        st.error(f"⚠️ Model file '{MODEL_PATH}' not found. Please train the model first.")
    else:
        # File uploader
        uploaded_file = st.file_uploader(
            "UPLOAD X-RAY IMAGE", 
            type=["jpg", "jpeg", "png"],
        )
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                img = Image.open(uploaded_file)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(img, caption='Input Radiograph', use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Preprocess the image
                img_size = (180, 180)
                img_resized = img.convert('RGB').resize(img_size)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Analyze button
                if st.button("RUN DIAGNOSTIC ANALYSIS"):
                    with st.spinner('Processing image data...'):
                        # Make prediction
                        prediction = model.predict(img_array, verbose=0)[0][0]
                        
                        # Display result
                        if prediction > 0.5:
                            label = "PNEUMONIA DETECTED"
                            confidence = prediction * 100
                            st.markdown(
                                f'<div class="prediction-box pneumonia">'
                                f'<div class="result-title">⚠️ {label}</div>'
                                f'<div class="confidence-text">Confidence: {confidence:.2f}%</div>'
                                f'</div>', 
                                unsafe_allow_html=True
                            )
                        else:
                            label = "NORMAL LUNG"
                            confidence = (1 - prediction) * 100
                            st.markdown(
                                f'<div class="prediction-box normal">'
                                f'<div class="result-title">✅ {label}</div>'
                                f'<div class="confidence-text">Confidence: {confidence:.2f}%</div>'
                                f'</div>', 
                                unsafe_allow_html=True
                            )
                        
                        # Additional stats
                        st.markdown("<br>", unsafe_allow_html=True)
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        
                        with stat_col1:
                            st.markdown(
                                f'<div class="stat-card">'
                                f'<div class="stat-number">{confidence:.1f}%</div>'
                                f'<div class="stat-label">Confidence</div>'
                                f'</div>', 
                                unsafe_allow_html=True
                            )
                        
                        with stat_col2:
                            st.markdown(
                                f'<div class="stat-card">'
                                f'<div class="stat-number">CNN</div>'
                                f'<div class="stat-label">Model</div>'
                                f'</div>', 
                                unsafe_allow_html=True
                            )
                        
                        with stat_col3:
                            st.markdown(
                                f'<div class="stat-card">'
                                f'<div class="stat-number">High</div>'
                                f'<div class="stat-label">Precision</div>'
                                f'</div>', 
                                unsafe_allow_html=True
                            )

            except Exception as e:
                if uploaded_file.name.startswith("._"):
                    st.warning(f"⚠️ **Invalid File Selected:** You uploaded `{uploaded_file.name}`")
                    st.error("🛑 This is a hidden macOS system file (metadata), not the real image.")
                    st.info(f"👉 Please find the file named **`{uploaded_file.name[2:]}`** instead.")
                else:
                    st.error(f"Error loading image: {str(e)}")
                    st.info("The uploaded file seems to be corrupted or not a valid image. Please try uploading a different JPEG or PNG file.")

        else:
            st.info("Waiting for input image...")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer info
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem;">'
    'AUTOMATED DIAGNOSTIC ASSISTANT V1.0'
    '</div>', 
    unsafe_allow_html=True
)
