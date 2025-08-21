import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# --- Custom Theme (with .result-box style removed) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
    }
    .big-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #00FF9D;
    }
    /* The .result-box style is no longer needed */
    div.stMarkdown p {
        color: #E0E0E0;
    }
    div.stMarkdown h1 {
        color: #00FF9D;
    }
    div.stMarkdown h2, div.stMarkdown h3, div.stMarkdown h4 {
        color: #00D4FF;
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
IMG_SIZE = 224
CLASS_NAMES = {
    "glioma": "A tumor that develops in the glial cells of the brain",
    "meningioma": "A tumor that forms in the meninges",
    "notumor": "No Tumor - Healthy brain tissue",
    "pituitary": "A tumor in the pituitary gland"
}


# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('final_brain_tumor_cnn.keras')
        st.success("Neural Network Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Preprocessing ---
def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- Streamlit App Interface ---
st.title("Brain Tumor Classification")
st.markdown("### AI-Powered MRI Analysis System")
st.markdown("""
This application uses a sophisticated Neural Network to analyze brain MRI scans 
and detect various types of brain tumors. Simply upload an MRI image to get started.
""")

st.markdown("### Upload Your MRI Scan")

st.markdown("""
<div style="color: #00D4FF; font-size: 0.8em; margin-bottom: 5px;">
    Supported formats: JPG, JPEG, PNG | For best results, use clear MRI scans
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### Uploaded MRI Scan")
    st.image(image, caption='', use_column_width=True)
    
    if model:
        with st.spinner('Analyzing the MRI scan...'):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img, verbose=0)
            
            # Get prediction
            pred_index = np.argmax(prediction)
            pred_class = list(CLASS_NAMES.keys())[pred_index]
            confidence = np.max(prediction) * 100
            
            st.markdown("### Analysis Results")
            # --- CHANGE: The line below that created the box has been removed ---
            # st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            # Display main prediction
            st.markdown("#### Primary Diagnosis")
            st.markdown(f'<p class="big-font">{pred_class.title()}</p>', unsafe_allow_html=True)
            st.markdown(f"*Confidence: {confidence:.1f}%*")
            
            # Show description
            st.markdown("#### Description")
            st.markdown(f"{CLASS_NAMES[pred_class]}")
            
            # Show probability distribution
            st.markdown("#### Detailed Analysis")
            prob_df = pd.DataFrame({
                'Tumor Type': CLASS_NAMES.keys(),
                'Probability (%)': [p * 100 for p in prediction[0]]
            })
            st.bar_chart(prob_df.set_index('Tumor Type'), height=400)
            
            # --- CHANGE: The line below that closed the box has been removed ---
            # st.markdown('</div>', unsafe_allow_html=True)
            
            # Medical disclaimer
            st.markdown("""
                <div style="padding: 20px; border-radius: 10px; background-color: #1C3A4E; border: 1px solid #264B63; color: #E0E0E0; margin-top: 20px;">
                ⚕️ This analysis is provided as a supportive tool and should not be used as the sole basis for medical decisions. 
                Please consult with a qualified healthcare professional for proper medical diagnosis and treatment.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Model not available. Please check if the model file exists and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.8em; margin-top: 40px;">
    Brain Tumor Classification System | Powered by TensorFlow & Streamlit<br>
    For educational and research purposes only
</div>
""", unsafe_allow_html=True)