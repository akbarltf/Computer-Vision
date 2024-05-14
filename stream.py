import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import cv2
import io

# Function to handle file upload
def file_upload():
    uploaded_file = st.file_uploader("Upload File", type=["jpg", "mp4"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]  # Get file type
        st.write(f"{file_type} berhasil diupload")

        if file_type == "jpg":
            # Display the button to show results for image
            if st.button("Tampilkan Hasil Deteksi Gambar"):
                process_image(uploaded_file)
        elif file_type == "mp4":
            # Display the button to process the video
            if st.button("Proses Video"):
                show_video_results_button = st.empty()
                show_video_results_button.text("Video sedang diproses...")
                process_video(uploaded_file, show_video_results_button)

# Function to process uploaded image
def process_image(uploaded_file):
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Read the uploaded image as numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform inference on the uploaded image
    results = model(img, classes=0)  # Specify the "person" class index

    # Define class names
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # Filter results for the "person" class
    person_count = 0
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        cls = boxes.cls.tolist()  # Convert tensor to list
        for class_index in cls:
            class_name = class_names[int(class_index)]
            if class_name == 'person':
                person_count += 1

    # Draw bounding boxes on the image
    for result in results:
        if result.names[0] == 'person':
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the total count of detected people
    cv2.putText(img, f"Jumlah orang: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert image to bytes for display
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    # Display the image with bounding boxes and person count
    st.image(img_bytes, channels="BGR")

    # Download the processed image
    st.download_button(label="Download Hasil Deteksi Gambar", data=img_bytes, file_name="detected_image.jpg", mime="image/jpeg")

# Function to process uploaded video
def process_video(uploaded_file, show_video_results_button):
    # Save uploaded file to a temporary location
    temp_location = tempfile.NamedTemporaryFile(delete=False)
    temp_location.write(uploaded_file.read())
    temp_location.close()  # Close the file

    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Open the video file
    cap = cv2.VideoCapture(temp_location.name)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object to write output video
    output_video_path = "detected_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_video_path, fourcc, frame_fps, (frame_width, frame_height))

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # Initialize variables to store total people count in the video
    total_people = 0

    # Loop through each frame in the video
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Perform object detection on the frame
            results = model(frame, classes=0)  # Specify the "person" class index

            # Filter results for the "person" class
            frame_people_count = 0
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                cls = boxes.cls.tolist()  # Convert tensor to list
                for class_index in cls:
                    class_name = class_names[int(class_index)]
                    if class_name == 'person':
                        frame_people_count += 1

            # Update total people count
            total_people += frame_people_count

            # Draw bounding boxes on the frame
            for result in results:
                if result.names[0] == 'person':
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw text on the frame indicating the number of people detected in the current frame
            cv2.putText(frame, f"Jumlah orang: {frame_people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the frame to the output video
            output_video.write(frame)

        else:
            break

    # Release the video capture and output video objects
    cap.release()
    output_video.release()

    # Remove temporary file
    os.unlink(temp_location.name)

    # Hide the "Proses Video..." message
    show_video_results_button.empty()

    st.write(f"Video selesai dideteksi!")

    # Download the processed video
    with open(output_video_path, "rb") as file:
        video_bytes = file.read()
    st.download_button(label="Download Hasil Deteksi Video", data=video_bytes, file_name="detected_video.mp4", mime="video/mp4")

# Main code
st.title("Deteksi dan Perhitungan Manusia Menggunakan Model YOLOv8")

# Upload image or video
st.header("Upload Gambar atau Video")
file_upload()
