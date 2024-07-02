import streamlit as st
import cv2
from ultralytics import YOLO
from twilio.rest import Client
import os
import imutils
import numpy as np
import requests
import pygame

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}

    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'

        if v > 1:
            prediction_text += 's'

        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

# Function to send SMS using Twilio
def send_sms(message):
    # Twilio credentials
    # login to twilio and get the account_sid, auth_token, twilio_phone_number, to_phone_number
    # this credentails is to send sms provide the respective credential 
    account_sid = '' 
    auth_token = ''
    twilio_phone_number = ''
    to_phone_number = ''

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=to_phone_number
    )

# Function for motion detection
def motion_detection(cap):
    pygame.init()
    pygame.mixer.music.load("") # provide the buzzer.mp3 file location here
    # Twilio credentials
    # login to twilio and get the account_sid, auth_token, twilio_phone_number, to_phone_number
    # this credentails is to send photo thourh whatsapp provide the respective credential
    account_sid = ""
    auth_token = ""
    twilio_phone_number = "whatsapp:" # in credentials the number should be whatsapp number
    your_phone_number = "whatsapp:"

    #imgbb api key
    #login to imgbb website and get the imgbb api key
    imgbb_api_key = ""

    client = Client(account_sid, auth_token)
    MOVEMENT_DETECTED_PERSISTENCE = 100

    first_frame = None
    next_frame = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0
    first_frame_sent = False

    while True:
        ret, frame = cap.read()
        text = "Unoccupied"

        if not ret:
            print("CAPTURE ERROR")
            continue

        frame = imutils.resize(frame, width=750)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray
            first_frame_filename = "first_frame.jpg"
            cv2.imwrite(first_frame_filename, frame)

            with open(first_frame_filename, "rb") as file:
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    files={"image": file},
                    data={"key": imgbb_api_key}
                )
                result = response.json()
                img_url = result["data"]["url"]

            message = client.messages.create(
                body="Motion Detected!! Counter:100",
                from_=twilio_phone_number,
                to=your_phone_number,
                media_url=[img_url]
            )

            first_frame_sent = True

        delay_counter += 1

        if delay_counter > 10:
            delay_counter = 0
            first_frame = next_frame

        next_frame = gray
        frame_delta = cv2.absdiff(first_frame, next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        transient_movement_flag = False

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if cv2.contourArea(c) > 2000:
                transient_movement_flag = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if transient_movement_flag:
            movement_persistent_flag = True
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE
            pygame.mixer.music.play()

        if movement_persistent_counter > 0:
            text = "Movement Detected " + str(movement_persistent_counter)
            movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"

        cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        cv2.imshow("frame", np.hstack((frame_delta, frame)))

        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break

# Streamlit app
def main():
    st.set_page_config(
        page_title="Fire and Motion Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",
    )
    st.title("Fire and Motion Detection")

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Detection Type", ("Fire Detection", "Motion Detection"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]

    with col2:
        selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    # Add a section divider
    st.markdown("---")

    # Set confidence and IOU thresholds for fire detection
    if model_type == "Fire Detection":
        col1, col2 = st.columns(2)
        with col2:
            conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
            with st.expander("What is Confidence Threshold?"):
                st.caption("The Confidence Threshold is a value between 0 and 1.")
                st.caption("It determines the minimum confidence level required for an object detection.")
                st.caption("If the confidence of a detected object is below this threshold, it will be ignored.")
                st.caption("You can adjust this threshold to control the number of detected objects.")
                st.caption("Lower values make the detection more strict, while higher values allow more detections.")
        with col1:
            iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)
            with st.expander("What is IOU Threshold?"):
                st.caption("The IOU (Intersection over Union) Threshold is a value between 0 and 1.")
                st.caption("It determines the minimum overlap required between the predicted bounding box")
                st.caption("and the ground truth box for them to be considered a match.")
                st.caption("You can adjust this threshold to control the precision and recall of the detections.")
                st.caption("Higher values make the matching more strict, while lower values allow more matches.")

        # Add a section divider
        st.markdown("---")

        # Option to use live camera input for fire detection
        if st.checkbox("Use Live Camera Input for Fire Detection"):
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Error: Unable to open the camera.")
                return

            image_container = st.empty()
            sms_sent = False

            try:
                while True:
                    ret, frame = cap.read()

                    if not ret:
                        st.error("Error: Unable to read frame from the camera.")
                        break

                    prediction, text = predict_image(model, frame, conf_threshold, iou_threshold)

                    combined_image = cv2.hconcat([frame, prediction])
                    image_container.image(combined_image, channels="BGR",
                                          caption="Live Camera Feed and Prediction", use_column_width=True)
                    st.success(text)

                    if "fire" in text.lower() and not sms_sent:
                        send_sms("Fire detected! on camera 1")
                        sms_sent = True

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cap.release()

    # Option to use live camera input for motion detection
    elif st.checkbox("Use Live Camera Input for Motion Detection"):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Unable to open the camera.")
            return

        try:
            motion_detection(cap)
        finally:
            cap.release()

if __name__ == "__main__":
    main()
