
---

# Automated Smart Home

This project implements an Automated Smart Home system with advanced features such as Face Recognition, Motion Detection, and Fire Detection. The system leverages AI, deep learning, and computer vision technologies to enhance home security and safety.

## Features

### Face Recognition
- Enhances security by recognizing authorized individuals.
- Uses OpenCV for real-time face recognition.

### Motion Detection
- Provides real-time alerts when motion is detected.
- Sends an image of the detected motion through WhatsApp using Twilio.
- Converts the motion detection image to a link using the imgbb API.

### Fire Detection
- Ensures swift identification of potential hazards.
- Uses a pre-trained YOLOv8 Machine Learning model for fire detection.
- Sends real-time SMS notifications to the user using Twilio.

### Multiple Devices
- Connects multiple phones as cameras within a local network using the IPWebcam app.

## File Descriptions

- **combinedcode.py**: Main script that combines face recognition, motion detection, and fire detection functionalities.
- **multipledevice.py**: Script to connect multiple devices as cameras using the IPWebcam app.
- **buzzer.mp3**: Sound file that plays when motion is detected.


### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/automated-smart-home.git
   cd automated-smart-home
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Face Recognition, Motion Detection, and Fire Detection**:
   
   Run the `combinedcode.py` script to start the integrated functionalities of face recognition, motion detection, and fire detection.

   ```bash
   python combinedcode.py
   ```

2. **Connecting Multiple Devices**:
   
   Use the `multipledevice.py` script to connect multiple devices as cameras.

   ```bash
   python multipledevice.py
   ```

3. **Running Fire Detection with Streamlit**:
   
   Run the fire detection component using Streamlit for a web-based interface.

   ```bash
   streamlit run combinedcode.py
   ```

## Usage

### Face Recognition

- The system will recognize faces in real-time and grant or deny access based on the recognized faces.

### Motion Detection

- Upon detecting motion, the system will send a real-time alert via WhatsApp with an image of the detected motion.
- The image is uploaded to imgbb, and the link is sent through Twilio WhatsApp API.

### Fire Detection

- The system continuously monitors for fire and sends an SMS notification immediately upon detecting fire using Twilio SMS API.

### Multiple Devices

- Use the IPWebcam app to connect multiple phones as cameras. The `multipledevice.py` script handles the connection and stream management.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [YOLOv8](https://github.com/ultralytics/yolov8)
- [Twilio](https://www.twilio.com/)
- [IPWebcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en&gl=US)
- [Streamlit](https://streamlit.io/)

---
