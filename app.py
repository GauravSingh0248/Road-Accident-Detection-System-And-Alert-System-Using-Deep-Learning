import streamlit as st
from accident_detection import load_model, predict_image
from twilio.rest import Client
import cv2
from PIL import Image
import numpy as np

# TWILIO_SID = ''
# TWILIO_AUTH_TOKEN = ''
# TWILIO_PHONE = '+'
# TARGET_PHONE = '+'

model = load_model('weights/model_weights.weights.h5')


def send_alert():
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body="🚨 Road Accident Detected! Immediate attention required.",
        from_=TWILIO_PHONE,
        to=TARGET_PHONE
    )
    return message.sid

# Updated predict_image to accept PIL image instead of file path
def predict_image(model, pil_img):
    IMG_SIZE = 224  # Ensure this matches your model input size
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0][0]
    label = "Accident" if pred >= 0.5 else "No Accident"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, confidence

# Streamlit UI
st.title("🚦 Road Accident Detection & Alert System")
mode = st.radio("Choose Mode:", ["📷 Live Camera", "🖼️ Upload Images"])

# ----------- Live Camera Mode -----------
if mode == "📷 Live Camera":
    st.info("Capture a frame using webcam")
    if st.button("Capture"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert OpenCV BGR frame to PIL RGB image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            st.image(frame_rgb, caption="Captured Frame")

            label, confidence = predict_image(model, pil_img)
            st.write(f"Prediction: **{label}** with confidence {confidence:.2f}")

            if label == "Accident":
                st.error("🚨 Accident Detected!")
                sid = send_alert()
                st.success(f"✅ Alert sent (SID: {sid})")
            else:
                st.success("✅ No Accident Detected")
        else:
            st.warning("Could not access webcam")

# ----------- Upload Image Mode -----------
else:
    files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if files:
        for file in files:
            try:
                img = Image.open(file).convert("RGB")
            except Exception as e:
                st.error(f"Error loading image {file.name}: {e}")
                continue

            st.image(img, caption=file.name, use_container_width=True)
            label, confidence = predict_image(model, img)
            st.write(f"Prediction: **{label}** with confidence {confidence:.2f}")

            if label == "Accident":
                st.error("🚨 Accident Detected!")
                sid = send_alert()
                st.success(f"✅ Alert sent (SID: {sid})")
            else:
                st.success("✅ No Accident Detected")
