import streamlit as st
from accident_detection import load_model, predict_image
from twilio.rest import Client
import cv2
from PIL import Image
import numpy as np


# TWILIO_SID = 
# TWILIO_AUTH_TOKEN = 
# TWILIO_PHONE =
# TARGET_PHONE = 


model = load_model('weights/model_weights.h5')


def send_alert():
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body="🚨 Road Accident Detected! Immediate attention required.",
        from_=TWILIO_PHONE,
        to=TARGET_PHONE
    )
    return message.sid

# 🌐 Streamlit UI
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
            img_path = "temp_live.jpg"
            cv2.imwrite(img_path, frame)
            st.image(frame, channels="BGR", caption="Captured Frame")

            score = predict_image(model, img_path)
            st.write(f"Prediction Score: {score:.2f}")
            if score >= 0.5:
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
            img = Image.open(file).convert("RGB")
            img_path = f"temp_{file.name}"
            img.save(img_path)

            st.image(img, caption=file.name, use_column_width=True)
            score = predict_image(model, img_path)
            st.write(f"Prediction Score: {score:.2f}")

            if score >= 0.5:
                st.error("🚨 Accident Detected!")
                sid = send_alert()
                st.success(f"✅ Alert sent (SID: {sid})")
            else:
                st.success("✅ No Accident Detected")
