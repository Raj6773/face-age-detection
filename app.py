import streamlit as st
import cv2
import numpy as np

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

def detect_age(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_blob = cv2.dnn.blobFromImage(image[y:y+h, x:x+w], 1.0, (227, 227), (78.426, 87.769, 114.895), swapRB=False)
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, age, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image

# Streamlit UI
st.title("Face Age Detection - Upload or Webcam")
option = st.radio("Choose an option:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        result = detect_age(image)
        st.image(result, channels="BGR")

elif option == "Use Webcam":
    st.write("Starting webcam...")
    cap = cv2.VideoCapture(0)

    FRAME_WINDOW = st.image([])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_age(frame)
        FRAME_WINDOW.image(result, channels="BGR")

    cap.release()
