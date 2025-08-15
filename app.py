
import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile
import time
import random

from solvers.text_captcha import solve_text_captcha
from solvers.math_captcha import solve_math_captcha
from solvers.object_captcha import detect_objects
from solvers.gemini import gemini_solve_math

st.title(" CAPTCHA Solver Dashboard")
st.write("Upload CAPTCHA images, select type, and watch the magic happen ✨")

captcha_type = st.selectbox(
    "Select CAPTCHA type:",
    ("Text CAPTCHA", "Math CAPTCHA", "Object CAPTCHA")
)

uploaded_files = st.file_uploader(
    "📤 Upload CAPTCHA images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

if uploaded_files:
    existing = {(f.name, f.size) for f in st.session_state.uploaded_files_list}
    for f in uploaded_files:
        if (f.name, f.size) not in existing:
            st.session_state.uploaded_files_list.append(f)
else:
    st.session_state.uploaded_files_list = []


def random_color():
    """Generate a random RGB color tuple."""
    return tuple(random.randint(50, 255) for _ in range(3))


def solve_and_get_data(image_path, filename, image):
    start_time = time.time()
    difficulty = None

    if captcha_type == "Text CAPTCHA":
        result = solve_text_captcha(image_path)
        detected_text = result.get('refined') or result.get('text') or "N/A"
        source = result.get('source', 'unknown')
        difficulty = result.get("difficulty")
        elapsed_time = time.time() - start_time

        return {
            "table": {
                "Detected Text": [detected_text],
                "Source": [source],
                "Difficulty Score": [f"{difficulty}/10" if difficulty else "N/A"]
            },
            "difficulty": difficulty,
            "image": image,
            "elapsed_time": elapsed_time
        }

    elif captcha_type == "Math CAPTCHA":
        result = solve_math_captcha(image_path)
        if not result.get("answer") or result.get("error"):
            expr = result.get("expression", "")
            result = gemini_solve_math(expr)
        expression = result.get('expression', 'N/A')
        answer = result.get('answer', 'N/A')
        source = result.get('source', 'unknown')
        difficulty = result.get("difficulty")
        elapsed_time = time.time() - start_time

        return {
            "table": {
                "Expression": [expression],
                "Answer": [answer],
                "Source": [source],
                "Difficulty Score": [f"{difficulty}/10" if difficulty else "N/A"]
            },
            "difficulty": difficulty,
            "image": image,
            "elapsed_time": elapsed_time
        }

    elif captcha_type == "Object CAPTCHA":
        result = detect_objects(image_path)
        objects = result.get("objects", [])
        difficulty = result.get("difficulty")

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()

        # Assign unique colors for each object label
        label_colors = {}
        table_rows = []

        for obj in objects:
            name = obj.get("name", "Unknown")
            confidence = obj.get("confidence")
            bbox = obj.get("bbox")

            if name not in label_colors:
                label_colors[name] = random_color()

            conf_str = f"{confidence:.3f}" if isinstance(confidence, float) else "N/A"
            table_rows.append({"Object": name, "Confidence": conf_str})

            if bbox:
                x1, y1, x2, y2 = bbox
                color = label_colors[name]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label_text = f"{name} ({conf_str})" if isinstance(confidence, float) else name

                # Keep label inside image bounds
                label_y = max(0, y1 - 12)
                draw.text((x1, label_y), label_text, fill=color, font=font)

        elapsed_time = time.time() - start_time

        return {
            "table": table_rows,
            "difficulty": difficulty,
            "image": annotated_image,
            "elapsed_time": elapsed_time
        }


if st.session_state.uploaded_files_list:
    for uploaded_file in st.session_state.uploaded_files_list:
        image = Image.open(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        result_data = solve_and_get_data(image_path, uploaded_file.name, image)

        col_img, col_res = st.columns([1, 1.5])

        with col_img:
            st.image(result_data["image"], caption=uploaded_file.name, use_container_width=True)
            st.markdown(
                f"<p style='color:green;'><b>Time Taken:</b> {result_data['elapsed_time']:.2f} seconds</p>",
                unsafe_allow_html=True
            )

        with col_res:
            st.markdown(f"### Results for {uploaded_file.name}")
            st.table(result_data["table"])
            if result_data["difficulty"] is not None:
                st.markdown(
                    f"<p style='font-size:16px; color:#F39C12;'><b>Difficulty Score:</b> {result_data['difficulty']}/10</p>",
                    unsafe_allow_html=True
                )
            st.markdown("---")