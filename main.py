from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PLANT_DISEASE_MODEL_PATH = BASE_DIR / "trained_plant_disease_model.keras"
FERTILIZER_MODEL_PATH = BASE_DIR / "fertikizer.pkl"
FERTILIZER_LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
CROP_MODEL_PATH = BASE_DIR / "Navis_Base.pkl"
HOME_IMAGE_PATH = BASE_DIR / "image.png"

# Model prediction function
def model_prediction(test_image):
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for disease diagnosis. Install it to enable this feature."
        ) from exc

    if not PLANT_DISEASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {PLANT_DISEASE_MODEL_PATH}")

    model = tf.keras.models.load_model(PLANT_DISEASE_MODEL_PATH)
    if hasattr(test_image, "seek"):
        try:
            test_image.seek(0)
        except Exception:
            pass
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def predict_fertilizer(temparature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type):

    # Load model and label encoder
    with open(FERTILIZER_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FERTILIZER_LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    # Define all possible columns as per training data
    columns = [
        'Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
        'Soil Type_Black', 'Soil Type_Clayey', 'Soil Type_Loamy', 'Soil Type_Red', 'Soil Type_Sandy',
        'Crop Type_Barley', 'Crop Type_Cotton', 'Crop Type_Ground Nuts', 'Crop Type_Maize',
        'Crop Type_Millets', 'Crop Type_Oil seeds', 'Crop Type_Paddy', 'Crop Type_Pulses',
        'Crop Type_Sugarcane', 'Crop Type_Tobacco', 'Crop Type_Wheat'
    ]

    # Prepare input dictionary
    input_dict = {col: [0] for col in columns}
    input_dict['Temparature'] = [temparature]
    input_dict['Humidity '] = [humidity]
    input_dict['Moisture'] = [moisture]
    input_dict['Nitrogen'] = [nitrogen]
    input_dict['Potassium'] = [potassium]
    input_dict['Phosphorous'] = [phosphorous]
    input_dict[f'Soil Type_{soil_type}'] = [True]
    input_dict[f'Crop Type_{crop_type}'] = [True]

    input_df = pd.DataFrame(input_dict)
    pred = model.predict(input_df)[0]
    return le.inverse_transform([pred])[0]

def crop_prediction(a, b, c, d, e, f, g):
    try:
        with open(CROP_MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        features = [[float(a), float(b), float(c), float(d), float(e), float(f), float(g)]]
        ##st.write("Model type:", type(model))
        ##st.write("Input features:", features)
        prediction = model.predict(features)
        return prediction[0]
    except Exception as ex:
        st.error(f"Prediction error: {ex}")
        return "Unknown"
# Sidebar
st.sidebar.markdown(
    "<h2 style='color:#228B22;'>🚜 AgriCare</h2>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<p style='color:#6B8E23;'>Your digital assistant for healthy crops</p>",
    unsafe_allow_html=True
)
app_mode = st.sidebar.radio("Navigate", ["🏡 Home", "🌾 Diagnose Disease", "🌱 Recommend Crop", "🌿 Suggest Fertilizer"])
# Main Page
if app_mode == "🏡 Home":
    st.markdown(
        "<h1 style='text-align: center; color: #228B22;'>🌱 AgriCare Plant Doctor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color:#556B2F;'>Empowering farmers with AI to keep crops healthy and fields green.</p>",
        unsafe_allow_html=True,
    )
    if HOME_IMAGE_PATH.exists():
        img = Image.open(HOME_IMAGE_PATH)
        st.image(img, use_container_width=True, caption="AgriCare Plant Doctor")
    else:
        st.warning(f"Home image not found: {HOME_IMAGE_PATH}")
    st.markdown(
        """
        <div style='background-color:#F0FFF1; padding:25px; border-radius:10px; color:#222;'>
            <h3 style='color:#228B22;'>About AgriCare Plant Doctor</h3>
            <p>
                <b>AgriCare Plant Doctor</b> is an AI-powered web application designed to assist farmers and gardeners in maintaining healthy crops and maximizing yield. 
                This project leverages advanced machine learning models to provide the following services:
            </p>
            <ul>
                <li><b>🌾 Diagnose Disease:</b> Upload a photo of your plant leaf to detect possible diseases using deep learning image analysis.</li>
                <li><b>🌱 Recommend Crop:</b> Enter soil and environmental parameters to receive suggestions for the most suitable crop to grow in your field.</li>
                <li><b>🌿 Suggest Fertilizer:</b> Get personalized fertilizer recommendations based on your soil, crop, and environmental conditions to boost productivity.</li>
            </ul>
            <p>
                Our goal is to empower the agricultural community with easy-to-use, reliable, and intelligent digital tools for better decision-making and sustainable farming.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif app_mode == "🌾 Diagnose Disease":
    st.markdown(
        "<h2 style='color: #228B22;'>🌾 Diagnose Your Crop</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#556B2F;'>Upload a photo of your plant leaf below. Our AI will help you identify possible diseases and keep your crops healthy!</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        test_image = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])
        if test_image:
            st.image(test_image, caption="Your Uploaded Leaf", use_container_width=True)
    with col2:
        if test_image:
            if st.button("🔍 Diagnose"):
                try:
                    with st.spinner("Analyzing your crop..."):
                        result_index = model_prediction(test_image)
                        disease = class_names[result_index]
                        st.markdown(
                            f"""
                            <div style='background-color:#E6FFE6; padding:20px; border-radius:10px;'>
                                <h3 style='color:#228B22;'>🩺 Diagnosis Result</h3>
                                <p style='font-size:20px; color:#000000;'><b>{disease.replace('_', ' ')}</b></p>
                                <p style='color:#6B8E23;'>For detailed advice, consult your local agricultural expert.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                except FileNotFoundError:
                    st.error(f"Disease model file not found: {PLANT_DISEASE_MODEL_PATH}")
                    st.info("Train/export the model (see `Training.ipynb`) to enable this feature.")
                except RuntimeError as e:
                    st.error(str(e))
                    st.info("Install TensorFlow to enable disease diagnosis.")
                except Exception:
                    st.error("Sorry, we couldn't analyze this image. Please try another photo.")
        else:
            st.info("Please upload a clear image of a plant leaf to start diagnosis.")
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8B4513; margin-top:30px;'>"
        "🌻 <i>Powered by AI for Farmers. Stay healthy, stay green!</i> 🌻"
        "</div>",
        unsafe_allow_html=True,
    )
elif app_mode == "🌱 Recommend Crop":
    st.markdown(
        "<h2 style='color: #228B22;'>🌱 Crop Recommendation</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#556B2F;'>Enter soil and environmental values to get the most suitable crop recommendation.</p>",
        unsafe_allow_html=True,
    )

    with st.form("recommendation_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
            P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
            K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)

        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=6000000.0, step=0.5)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=1000000.0, step=1.0)

        with col3:
            ph = st.number_input("pH Level", min_value=0.0, max_value=140000.0, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)

        submit_btn = st.form_submit_button("🌾 Recommend Crop")

    if submit_btn:
        try:
            with st.spinner("Processing recommendation..."):
                crop = crop_prediction(N, P, K, temperature, humidity, ph, rainfall)
                st.success(f"✅ Recommended Crop: **{str(crop).capitalize()}**")
        except Exception as e:
            st.error("Error occurred while predicting the crop. Please check input values or model path.")  
elif app_mode == "🌿 Suggest Fertilizer":
    st.markdown(
        "<h2 style='color: #228B22;'>🌿 Fertilizer Recommendation</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#556B2F;'>Input the soil and crop details to get the recommended fertilizer.</p>",
        unsafe_allow_html=True,
    )

    with st.form("fertilizer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.5)
            Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
            Moisture = st.number_input("Moisture Level (%)", min_value=0.0, max_value=100.0, step=1.0)

        with col2:
            Soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
            Crop_Type = st.selectbox("Crop Type", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Millets", "Paddy", "Oil seeds", "Pulses", "Barley", "Ground Nuts", "Wheat"])

        with col3:
            N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
            P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
            K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)

        submit_fert = st.form_submit_button("🌿 Suggest Fertilizer")

    if submit_fert:
        try:
            with st.spinner("Analyzing soil and crop..."):
                fertilizer = predict_fertilizer(Temp, Humidity, Moisture, N, K, P, Soil_type, Crop_Type)
                st.success(f"✅ Recommended Fertilizer: **{str(fertilizer).upper()}**")
        except Exception as e:
            st.error(f"Error during fertilizer prediction: {e}")
