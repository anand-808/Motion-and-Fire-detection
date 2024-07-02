import streamlit as st
import cv2
from ultralytics import YOLO
import os
from twilio.rest import Client

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    # Predict objects using the model
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}

    # Count the number of occurrences for each class
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'

        if v > 1:
            prediction_text += 's'

        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    # Convert the result image to RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

# Function to send SMS using Twilio

def send_sms(message):
    # Twilio credentials
    # login to twilio and get the account_sid, auth_token, twilio_phone_number, to_phone_number
    account_sid = ''
    auth_token = ''
    twilio_phone_number = ''
    to_phone_number = ''

    client = Client(account_sid, auth_token)
    for i in range(5):
        message = client.messages.create(
            body=message,  # Use the provided message argument
            from_=twilio_phone_number,
            to=to_phone_number
        )


# Function to open cameras
def open_cameras(urls):
    caps = []
    for url in urls:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set a higher frame rate, adjust as needed

        if not cap.isOpened():
            st.error(f"Error: Unable to open camera with URL: {url}. Error code: {cv2.CAP_PROP_POS_FRAMES}")
        else:
            st.success(f"Camera opened successfully with URL: {url}")
            caps.append(cap)

    return caps

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Fire Detection",
        page_icon="🔥",
        initial_sidebar_state="collapsed",

    )
    st.title("Fire Detection with Multiple Inputs")

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]

    with col2:
        selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    # Load the selected model
    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    # Add a section divider
    st.markdown("---")

    # Set confidence and IOU thresholds
    col1, col2 = st.columns(2)
    with col2:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.70, 0.05)  # Increase the default value
        with st.expander("What is Confidence Threshold?"):
            st.caption("The Confidence Threshold is a value between 0 and 1.")
            st.caption("It determines the minimum confidence level required for an object detection.")
            st.caption("If the confidence of a detected object is below this threshold, it will be ignored.")
            st.caption("You can adjust this threshold to control the number of detected objects.")
            st.caption("Lower values make the detection more strict, while higher values allow more detections.")
    with col1:
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.7, 0.05)  # Increase the default value
        with st.expander("What is IOU Threshold?"):
            st.caption("The IOU (Intersection over Union) Threshold is a value between 0 and 1.")
            st.caption("It determines the minimum overlap required between the predicted bounding box")
            st.caption("and the ground truth box for them to be considered a match.")
            st.caption("You can adjust this threshold to control the precision and recall of the detections.")
            st.caption("Higher values make the matching more strict, while lower values allow more matches.")

    # Add a section divider
    st.markdown("---")

    # Option to use live camera input
    if st.checkbox("Use Live Camera Input"):
        # Get the URLs of the IP webcams
        camera_urls = [
            #go to playstore and download IP Webcam and start server then copy the IPv4 address and paste it here
            "http://192.168.1.41:8080/video", #"http://192.168.185.146:8080/video",  # Replace with the correct IP and port of your webcams
            # Add more URLs here if needed
        ]

        # OpenCV for continuous live camera detection
        caps = open_cameras(camera_urls)

        if not caps:
            st.error("No cameras were successfully opened.")
            return

        # Create empty containers for displaying images
        image_containers = [st.empty() for _ in range(len(caps))]

        # Placeholder to prevent automatic scrolling
        placeholder = st.empty()
        sms_sent = False

        while True:
            for i, cap in enumerate(caps):
                ret, frame = cap.read()

                if not ret:
                    st.error(f"Error: Unable to read frame from camera {i + 1}. Skipping...")
                    continue

                # Perform object detection on the frame
                prediction, text = predict_image(model, frame, conf_threshold, iou_threshold)

                # Display the live camera feed and prediction in Streamlit
                combined_image = cv2.hconcat([frame, prediction])
                image_containers[i].image(combined_image, channels="BGR",
                                          caption=f"Camera {i + 1} Feed and Prediction",
                                          use_column_width=True)  # Adjust width to fit the column

                st.success(text)

                # Check if fire is detected and send SMS using Twilio
                if "fire" in text.lower() and not sms_sent:
                    send_sms("Fire detected! on camera 1")
                    sms_sent = True

                # Manually trigger a rerun to update the placeholders without scrolling
            #st.experimental_rerun()

                    # Update the placeholder to reset it for the next iteration
            placeholder.empty()
if __name__ == "__main__":
    main()