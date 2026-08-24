import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import tempfile
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="YOLO Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9aa4b2;
    margin-bottom: 30px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.info-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

.feature-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 15px;
    padding: 22px;
    min-height: 130px;
}

.feature-title {
    font-size: 20px;
    font-weight: 700;
}

.feature-text {
    color: #9aa4b2;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


try:
    model = load_model()
except Exception as e:
    st.error("Could not load YOLO model.")
    st.error(str(e))
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎯 YOLO Object Detection & Tracking</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered object detection, tracking, counting and video analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Project Controls")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "🏠 Dashboard",
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📊 About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Technology Stack")

st.sidebar.write("• Python")
st.sidebar.write("• YOLOv8")
st.sidebar.write("• Ultralytics")
st.sidebar.write("• OpenCV")
st.sidebar.write("• NumPy")
st.sidebar.write("• Streamlit")

st.sidebar.markdown("---")

st.sidebar.info(
    "YOLOv8 Nano is used for fast object detection "
    "and tracking."
)


# =========================================================
# DASHBOARD
# =========================================================

if mode == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This system uses YOLOv8 computer vision to detect, "
        "track and count objects in images and videos."
    )

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Detection System Statistics</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Detection",
            value="Active"
        )

    with col2:
        st.metric(
            label="🆔 Tracking",
            value="Enabled"
        )

    with col3:
        st.metric(
            label="🔢 Counting",
            value="Enabled"
        )

    with col4:
        st.metric(
            label="🎥 Recording",
            value="Enabled"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # PROJECT FEATURES
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🚀 Key Features</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🖼️ Object Detection</div>
            <br>
            <div class="feature-text">
                Detect objects in images using YOLOv8.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🆔 Object Tracking</div>
            <br>
            <div class="feature-text">
                Track objects across video frames using IDs.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔢 Object Counting</div>
            <br>
            <div class="feature-text">
                Count detected objects by category.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎥 Video Analysis</div>
            <br>
            <div class="feature-text">
                Process uploaded videos with AI detection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📹 Video Recording</div>
            <br>
            <div class="feature-text">
                Save processed detection results as video.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Analytics</div>
            <br>
            <div class="feature-text">
                Display object counts and detection statistics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🤖 Model Information</div>',
        unsafe_allow_html=True
    )

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric(
            "Model",
            "YOLOv8 Nano"
        )

    with info2:
        st.metric(
            "Framework",
            "Ultralytics"
        )

    with info3:
        st.metric(
            "Computer Vision",
            "OpenCV"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # PROJECT FILES
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📁 Project Structure</div>',
        unsafe_allow_html=True
    )

    st.code("""
Object_Detection_Tracking/
│
├── images/
│   ├── test.jpg
│   └── test2.jpg
│
├── detect_image.py
├── webcam.py
├── tracking.py
├── test_yolo.py
├── app.py
│
├── yolov8n.pt
├── yolov8nseg.pt
│
└── output_tracking.mp4
    """)

    st.success(
        "✅ YOLO Object Detection & Tracking System is ready."
    )


# =========================================================
# IMAGE DETECTION
# =========================================================

elif mode == "🖼️ Image Detection":

    st.markdown(
        '<div class="section-title">🖼️ Image Object Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an image and YOLO will automatically detect "
        "objects in it."
    )

    uploaded_file = st.file_uploader(
        "📁 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        image_array = np.array(image)

        with st.spinner("🔍 Detecting objects..."):

            results = model(
                image_array,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            detected_names = []

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_id = int(cls)

                    detected_names.append(
                        model.names[class_id]
                    )

            counts = Counter(detected_names)

        # -------------------------------------------------
        # IMAGES
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📷 Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("🎯 Detection Result")

            st.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        st.markdown("---")

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        st.subheader("📊 Detection Statistics")

        stat1, stat2, stat3 = st.columns(3)

        with stat1:
            st.metric(
                "Total Objects",
                len(detected_names)
            )

        with stat2:
            st.metric(
                "Object Types",
                len(counts)
            )

        with stat3:
            st.metric(
                "Model",
                "YOLOv8"
            )

        # -------------------------------------------------
        # OBJECT COUNTS
        # -------------------------------------------------

        if counts:

            st.subheader("🔢 Object Count")

            count_columns = st.columns(
                min(len(counts), 4)
            )

            for index, (name, count) in enumerate(
                counts.items()
            ):

                with count_columns[
                    index % len(count_columns)
                ]:

                    st.metric(
                        name.capitalize(),
                        count
                    )

        else:

            st.warning(
                "No recognizable objects were detected."
            )


# =========================================================
# VIDEO DETECTION
# =========================================================

elif mode == "🎥 Video Detection":

    st.markdown(
        '<div class="section-title">'
        '🎥 Video Object Detection & Tracking'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload an MP4, AVI or MOV video to process it "
        "using YOLO."
    )

    uploaded_video = st.file_uploader(
        "📁 Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # -------------------------------------------------
        # SAVE INPUT VIDEO
        # -------------------------------------------------

        input_suffix = os.path.splitext(
            uploaded_video.name
        )[1]

        if input_suffix == "":
            input_suffix = ".mp4"

        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=input_suffix
        )

        video_data = uploaded_video.read()

        temp_input.write(video_data)

        temp_input.close()

        # Reset uploaded file
        uploaded_video.seek(0)

        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        cap = cv2.VideoCapture(
            temp_input.name
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open the uploaded video."
            )

            os.unlink(temp_input.name)

            st.stop()

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 20.0

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        # -------------------------------------------------
        # OUTPUT FILE
        # -------------------------------------------------

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        # Use mp4v for broad OpenCV compatibility
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not out.isOpened():

            cap.release()

            os.unlink(temp_input.name)
            os.unlink(output_path)

            st.error(
                "❌ Could not create output video."
            )

            st.stop()

        # -------------------------------------------------
        # PROCESSING UI
        # -------------------------------------------------

        st.subheader("⚙️ Processing Video")

        progress = st.progress(0)

        status = st.empty()

        processed = 0

        total_detected = 0

        all_detected_classes = []

        # -------------------------------------------------
        # PROCESS VIDEO
        # -------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            # Collect detected objects
            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_id = int(cls)

                    class_name = model.names[
                        class_id
                    ]

                    all_detected_classes.append(
                        class_name
                    )

                    total_detected += 1

            out.write(annotated)

            processed += 1

            if frame_count > 0:

                progress.progress(
                    min(
                        processed / frame_count,
                        1.0
                    )
                )

            status.text(
                f"Processing frame "
                f"{processed}/{frame_count}"
            )

        # -------------------------------------------------
        # RELEASE VIDEO
        # -------------------------------------------------

        cap.release()
        out.release()

        progress.progress(1.0)

        status.success(
            "✅ Video processing completed!"
        )

        # -------------------------------------------------
        # VIDEO INFORMATION
        # -------------------------------------------------

        st.markdown("---")

        st.subheader("📊 Video Statistics")

        stat1, stat2, stat3, stat4 = st.columns(4)

        with stat1:
            st.metric(
                "Frames",
                processed
            )

        with stat2:
            st.metric(
                "FPS",
                f"{fps:.1f}"
            )

        with stat3:
            st.metric(
                "Resolution",
                f"{width} × {height}"
            )

        with stat4:
            st.metric(
                "Detected Objects",
                total_detected
            )

        # -------------------------------------------------
        # OBJECT COUNTS
        # -------------------------------------------------

        video_counts = Counter(
            all_detected_classes
        )

        if video_counts:

            st.subheader(
                "🔢 Detected Object Categories"
            )

            count_columns = st.columns(
                min(len(video_counts), 4)
            )

            for index, (
                object_name,
                count
            ) in enumerate(
                video_counts.items()
            ):

                with count_columns[
                    index % len(count_columns)
                ]:

                    st.metric(
                        object_name.capitalize(),
                        count
                    )

        # -------------------------------------------------
        # ORIGINAL VIDEO
        # -------------------------------------------------

        st.markdown("---")

        st.subheader("📹 Original Video")

        # Display original uploaded video
        st.video(
            video_data
        )

        # -------------------------------------------------
        # DETECTION VIDEO
        # -------------------------------------------------

        st.subheader(
            "🎯 Detection & Tracking Result"
        )

        with open(
            output_path,
            "rb"
        ) as video_file:

            output_video = video_file.read()

        st.video(
            output_video
        )

        # -------------------------------------------------
        # DOWNLOAD BUTTON
        # -------------------------------------------------

        st.download_button(
            label="⬇️ Download Detection Video",
            data=output_video,
            file_name="yolo_detection_output.mp4",
            mime="video/mp4"
        )

        # -------------------------------------------------
        # CLEAN TEMP INPUT
        # -------------------------------------------------

        try:
            os.unlink(
                temp_input.name
            )
        except:
            pass


# =========================================================
# ABOUT PROJECT
# =========================================================

elif mode == "📊 About Project":

    st.markdown(
        '<div class="section-title">'
        '📊 About This Project'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # OBJECTIVE
    # -----------------------------------------------------

    st.subheader("🎯 Objective")

    st.info(
        "The objective of this project is to develop an "
        "AI-powered computer vision system capable of "
        "detecting, tracking and counting objects in "
        "images and videos using YOLOv8."
    )

    # -----------------------------------------------------
    # TECHNOLOGIES
    # -----------------------------------------------------

    st.subheader("🛠️ Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
        **Programming**

        • Python  
        • NumPy  
        • OpenCV
        """)

    with tech2:
        st.markdown("""
        **AI / Machine Learning**

        • YOLOv8  
        • Ultralytics  
        • Computer Vision
        """)

    with tech3:
        st.markdown("""
        **Application**

        • Streamlit  
        • Web Dashboard  
        • Video Processing
        """)

    st.markdown("---")

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    st.subheader("🔍 Project Features")

    features = [
        "Object detection using YOLOv8",
        "Real-time webcam detection",
        "Object tracking with unique IDs",
        "Object counting",
        "Image detection",
        "Video detection",
        "Detection result recording",
        "Streamlit web dashboard",
        "Detection statistics"
    ]

    for feature in features:
        st.success(
            f"✅ {feature}"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # APPLICATIONS
    # -----------------------------------------------------

    st.subheader("💼 Real-World Applications")

    app1, app2 = st.columns(2)

    with app1:

        st.markdown("""
        **🏙️ Smart City**

        • Traffic monitoring  
        • Vehicle detection  
        • Crowd monitoring  
        • Road safety
        """)

        st.markdown("""
        **🏪 Retail**

        • Customer counting  
        • Store analytics  
        • Customer movement analysis
        """)

    with app2:

        st.markdown("""
        **🔐 Security**

        • Surveillance  
        • Intrusion detection  
        • People tracking
        """)

        st.markdown("""
        **🏭 Industry**

        • Object monitoring  
        • Safety monitoring  
        • Automated inspection
        """)

    st.markdown("---")

    # -----------------------------------------------------
    # PROJECT WORKFLOW
    # -----------------------------------------------------

    st.subheader("🔄 Project Workflow")

    st.code("""
Input
  ↓
Image / Video / Webcam
  ↓
OpenCV
  ↓
YOLOv8 Model
  ↓
Object Detection
  ↓
Object Tracking
  ↓
Object Counting
  ↓
Annotated Output
  ↓
Streamlit Dashboard
    """)

    st.markdown("---")

    st.success(
        "🎯 YOLO Object Detection & Tracking System "
        "successfully implemented."
    )