
import csv
import tempfile
import time
from pathlib import Path

import cv2
import easyocr
import streamlit as st
from ultralytics import YOLO




st.set_page_config(
    page_title="Vehicle Tracking + License Plate OCR",
    page_icon="🚗",
    layout="wide"
)

st.title("Vehicle Tracking + License Plate OCR")

st.markdown(
    """
    Upload an image or video to detect vehicles, detect license plates,
    and recognize plate text using EasyOCR.
    """
)




BASE_DIR = Path(__file__).resolve().parent

VEHICLE_MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"
PLATE_MODEL_PATH = BASE_DIR / "models" / "best.pt"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "Frame",
    "Track_ID",
    "Vehicle_Type",
    "License_Plate",
    "Vehicle_Confidence"
]




if not VEHICLE_MODEL_PATH.exists():
    st.error(f"Vehicle model not found: {VEHICLE_MODEL_PATH}")
    st.stop()

if not PLATE_MODEL_PATH.exists():
    st.error(f"Plate model not found: {PLATE_MODEL_PATH}")
    st.stop()




@st.cache_resource
def load_models(vehicle_path, plate_path):
    vehicle_model = YOLO(vehicle_path)
    plate_model = YOLO(plate_path)
    ocr_reader = easyocr.Reader(["ar", "en"], gpu=False)

    return vehicle_model, plate_model, ocr_reader


try:
    with st.spinner("Loading AI models..."):
        vehicle_model, plate_model, ocr_reader = load_models(
            str(VEHICLE_MODEL_PATH),
            str(PLATE_MODEL_PATH)
        )
    st.success("AI models loaded successfully.")

except Exception as error:
    st.error(f"Could not load models: {error}")
    st.stop()




def clean_plate_text(text):
    # Remove spaces and symbols while keeping letters and digits.
    return "".join(ch for ch in text if ch.isalnum()).strip()


def detect_and_read_plate(
    vehicle_crop,
    plate_model,
    ocr_reader,
    ocr_confidence
):
    # Return no result if the crop is empty.
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, None

    # Detect the license plate inside the vehicle crop.
    results = plate_model.predict(
        source=vehicle_crop,
        conf=0.25,
        verbose=False
    )

    if not results or results[0].boxes is None:
        return None, None

    if len(results[0].boxes) == 0:
        return None, None

    # Select the plate with the highest confidence.
    best_box = None
    best_confidence = 0.0

    for box in results[0].boxes:
        confidence = float(box.conf[0])

        if confidence > best_confidence:
            best_confidence = confidence
            best_box = tuple(
                map(int, box.xyxy[0].tolist())
            )

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = best_box
    height, width = vehicle_crop.shape[:2]

    # Keep plate coordinates inside the vehicle crop.
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None, best_box

    plate_crop = vehicle_crop[y1:y2, x1:x2]

    if plate_crop.size == 0:
        return None, best_box

    # Enlarge the plate to improve OCR readability.
    plate_crop = cv2.resize(
        plate_crop,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray_plate = cv2.cvtColor(
        plate_crop,
        cv2.COLOR_BGR2GRAY
    )

    # Recognize Arabic and English text.
    ocr_results = ocr_reader.readtext(
        gray_plate,
        detail=1,
        paragraph=False
    )

    recognized_parts = []

    for _, text, confidence in ocr_results:
        if confidence >= ocr_confidence:
            cleaned = clean_plate_text(text)

            if cleaned:
                recognized_parts.append(cleaned)

    plate_text = " ".join(recognized_parts)

    return (plate_text or None), best_box


def save_csv(records, csv_path):
    # Save results without using Pandas.
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS
        )
        writer.writeheader()
        writer.writerows(records)


def detect_image(
    image,
    vehicle_confidence,
    ocr_confidence
):
    # Detect vehicles in a single image.
    results = vehicle_model.predict(
        source=image,
        classes=[2, 3, 5, 7],
        conf=vehicle_confidence,
        verbose=False
    )

    annotated = image.copy()
    records = []

    if not results or results[0].boxes is None:
        return annotated, records

    boxes = results[0].boxes

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        height, width = image.shape[:2]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        vehicle_type = vehicle_model.names.get(
            class_id,
            str(class_id)
        )

        vehicle_crop = image[y1:y2, x1:x2]

        plate_text, plate_box = detect_and_read_plate(
            vehicle_crop,
            plate_model,
            ocr_reader,
            ocr_confidence
        )

        # A still image has no persistent tracking ID.
        display_text = plate_text or "Not recognized"
        label = f"Vehicle {index + 1} | {vehicle_type}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        if plate_box is not None:
            px1, py1, px2, py2 = plate_box

            # Convert plate coordinates to full-image coordinates.
            px1 += x1
            px2 += x1
            py1 += y1
            py2 += y1

            cv2.rectangle(
                annotated,
                (px1, py1),
                (px2, py2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                display_text,
                (px1, max(20, py1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        records.append({
            "Frame": 0,
            "Track_ID": f"Image-{index + 1}",
            "Vehicle_Type": vehicle_type,
            "License_Plate": display_text,
            "Vehicle_Confidence": round(confidence, 3)
        })

    return annotated, records




mode = st.radio(
    "Choose input type",
    ["Image", "Video"],
    horizontal=True
)

col1, col2 = st.columns(2)

with col1:
    vehicle_confidence = st.slider(
        "Vehicle Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

with col2:
    ocr_confidence = st.slider(
        "OCR Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05
    )




if mode == "Image":

    st.header("Image Detection and License Plate OCR")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="image_upload"
    )

    if uploaded_image is not None:
        file_bytes = uploaded_image.getvalue()

        # Decode the uploaded image using OpenCV.
        import numpy as np

        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error("Could not read this image file.")

        elif st.button("Analyze Image", type="primary"):

            with st.spinner(
                "Detecting vehicles and reading license plates..."
            ):
                try:
                    annotated_image, records = detect_image(
                        image,
                        vehicle_confidence,
                        ocr_confidence
                    )

                    run_id = str(int(time.time() * 1000))

                    image_output_path = OUTPUT_DIR / (
                        f"image_result_{run_id}.jpg"
                    )

                    csv_output_path = OUTPUT_DIR / (
                        f"image_results_{run_id}.csv"
                    )

                    if not cv2.imwrite(
                        str(image_output_path),
                        annotated_image
                    ):
                        raise RuntimeError(
                            "Could not save the processed image."
                        )

                    save_csv(records, csv_output_path)

                    st.success("Image analysis completed.")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Original Image")
                        st.image(
                            cv2.cvtColor(
                                image,
                                cv2.COLOR_BGR2RGB
                            ),
                            use_container_width=True
                        )

                    with col2:
                        st.subheader("Processed Image")
                        st.image(
                            cv2.cvtColor(
                                annotated_image,
                                cv2.COLOR_BGR2RGB
                            ),
                            use_container_width=True
                        )

                    st.subheader("Detection Summary")
                    metric1, metric2 = st.columns(2)

                    metric1.metric(
                        "Vehicles Detected",
                        len(records)
                    )

                    metric2.metric(
                        "Recognized Plates",
                        sum(
                            1 for record in records
                            if record["License_Plate"]
                            != "Not recognized"
                        )
                    )

                    st.subheader("Results")
                    if records:
                        st.dataframe(
                            records,
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            "No vehicles were detected in this image."
                        )

                    st.download_button(
                        "Download Processed Image",
                        data=image_output_path.read_bytes(),
                        file_name=image_output_path.name,
                        mime="image/jpeg"
                    )

                    st.download_button(
                        "Download CSV Report",
                        data=csv_output_path.read_bytes(),
                        file_name=csv_output_path.name,
                        mime="text/csv"
                    )

                except Exception as error:
                    st.error(f"Image processing failed: {error}")




else:

    st.header("Video Tracking and License Plate OCR")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    ocr_interval = st.selectbox(
        "Run OCR every N frames",
        options=[1, 2, 3, 5, 10],
        index=2
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Video Processing", type="primary"):

            run_id = str(int(time.time() * 1000))

            output_video_path = OUTPUT_DIR / (
                f"processed_{run_id}.mp4"
            )

            csv_output_path = OUTPUT_DIR / (
                f"video_results_{run_id}.csv"
            )

            input_video_path = None
            capture = None
            writer = None

            records = []
            last_plate_text = {}
            saved_readings = set()

            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_display = st.empty()

            frame_index = 0
            processing_error = None

            try:
                # Save the uploaded video to a temporary file.
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                ) as temporary_input:
                    temporary_input.write(
                        uploaded_video.getbuffer()
                    )
                    input_video_path = Path(
                        temporary_input.name
                    )

                capture = cv2.VideoCapture(
                    str(input_video_path)
                )

                if not capture.isOpened():
                    raise RuntimeError(
                        "Could not open the uploaded video."
                    )

                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_width = int(
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                )
                frame_height = int(
                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )
                total_frames = int(
                    capture.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if fps <= 0:
                    fps = 25.0

                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame_width, frame_height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create the output video."
                    )

                while True:
                    success, frame = capture.read()

                    if not success:
                        break

                    tracking_results = vehicle_model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[2, 3, 5, 7],
                        conf=vehicle_confidence,
                        verbose=False
                    )

                    annotated_frame = frame.copy()

                    if (
                        tracking_results
                        and tracking_results[0].boxes is not None
                    ):
                        boxes = tracking_results[0].boxes

                        if boxes.id is not None:
                            vehicle_boxes = (
                                boxes.xyxy.cpu().numpy()
                            )
                            track_ids = (
                                boxes.id.int().cpu().tolist()
                            )
                            class_ids = (
                                boxes.cls.int().cpu().tolist()
                            )
                            confidences = (
                                boxes.conf.cpu().tolist()
                            )

                            for (
                                vehicle_box,
                                track_id,
                                class_id,
                                vehicle_score
                            ) in zip(
                                vehicle_boxes,
                                track_ids,
                                class_ids,
                                confidences
                            ):
                                vx1, vy1, vx2, vy2 = map(
                                    int,
                                    vehicle_box
                                )

                                vx1 = max(
                                    0, min(vx1, frame_width - 1)
                                )
                                vx2 = max(
                                    0, min(vx2, frame_width)
                                )
                                vy1 = max(
                                    0, min(vy1, frame_height - 1)
                                )
                                vy2 = max(
                                    0, min(vy2, frame_height)
                                )

                                if vx2 <= vx1 or vy2 <= vy1:
                                    continue

                                vehicle_crop = frame[
                                    vy1:vy2,
                                    vx1:vx2
                                ]

                                vehicle_type = (
                                    vehicle_model.names.get(
                                        int(class_id),
                                        str(class_id)
                                    )
                                )

                                plate_text = last_plate_text.get(
                                    track_id,
                                    "Not recognized"
                                )

                                if frame_index % ocr_interval == 0:
                                    detected_text, plate_box = (
                                        detect_and_read_plate(
                                            vehicle_crop,
                                            plate_model,
                                            ocr_reader,
                                            ocr_confidence
                                        )
                                    )

                                    if plate_box is not None:
                                        px1, py1, px2, py2 = (
                                            plate_box
                                        )

                                        px1 += vx1
                                        px2 += vx1
                                        py1 += vy1
                                        py2 += vy1

                                        cv2.rectangle(
                                            annotated_frame,
                                            (px1, py1),
                                            (px2, py2),
                                            (0, 255, 255),
                                            2
                                        )

                                    if detected_text:
                                        plate_text = detected_text
                                        last_plate_text[
                                            track_id
                                        ] = detected_text

                                        reading_key = (
                                            track_id,
                                            detected_text
                                        )

                                        if reading_key not in saved_readings:
                                            records.append({
                                                "Frame": frame_index,
                                                "Track_ID": track_id,
                                                "Vehicle_Type": vehicle_type,
                                                "License_Plate": detected_text,
                                                "Vehicle_Confidence": round(
                                                    float(vehicle_score),
                                                    3
                                                )
                                            })
                                            saved_readings.add(
                                                reading_key
                                            )

                                cv2.rectangle(
                                    annotated_frame,
                                    (vx1, vy1),
                                    (vx2, vy2),
                                    (0, 255, 0),
                                    2
                                )

                                label = (
                                    f"ID: {track_id} | "
                                    f"{vehicle_type} | "
                                    f"Plate: {plate_text}"
                                )

                                cv2.putText(
                                    annotated_frame,
                                    label,
                                    (vx1, max(25, vy1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA
                                )

                    writer.write(annotated_frame)

                    if frame_index % 5 == 0:
                        frame_display.image(
                            cv2.cvtColor(
                                annotated_frame,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption=f"Processing frame {frame_index}",
                            use_container_width=True
                        )

                    frame_index += 1

                    if total_frames > 0:
                        progress_bar.progress(
                            min(
                                int(
                                    frame_index / total_frames * 100
                                ),
                                100
                            )
                        )

                    status_text.write(
                        f"Processed frames: {frame_index} / "
                        f"{total_frames if total_frames > 0 else 'Unknown'}"
                    )

            except Exception as error:
                processing_error = error

            finally:
                if capture is not None:
                    capture.release()

                if writer is not None:
                    writer.release()

                if input_video_path is not None:
                    input_video_path.unlink(missing_ok=True)

            if processing_error is not None:
                st.error(
                    f"Video processing failed: {processing_error}"
                )

            else:
                save_csv(records, csv_output_path)

                progress_bar.progress(100)
                status_text.success(
                    "Video processing completed."
                )

                st.subheader("Processed Video")

                if (
                    output_video_path.exists()
                    and output_video_path.stat().st_size > 0
                ):
                    st.video(str(output_video_path))

                    st.download_button(
                        "Download Processed Video",
                        data=output_video_path.read_bytes(),
                        file_name=output_video_path.name,
                        mime="video/mp4"
                    )
                else:
                    st.error(
                        "The output video could not be created."
                    )

                st.subheader("Detection Summary")

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Processed Frames",
                    frame_index
                )

                metric2.metric(
                    "Vehicle IDs with Readings",
                    len({
                        record["Track_ID"]
                        for record in records
                    })
                )

                metric3.metric(
                    "Saved Plate Readings",
                    len(records)
                )

                st.subheader("License Plate Results")

                if records:
                    st.dataframe(
                        records,
                        use_container_width=True
                    )
                else:
                    st.warning(
                        "No license plate text was recognized. "
                        "Try a clearer video or adjust the confidence."
                    )

                st.download_button(
                    "Download CSV Report",
                    data=csv_output_path.read_bytes(),
                    file_name=csv_output_path.name,
                    mime="text/csv"
                )

