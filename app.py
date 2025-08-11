import streamlit as st
from PIL import Image
import tempfile

from solvers.text_captcha import solve_text_captcha
from solvers.math_captcha import solve_math_captcha
from solvers.object_captcha import detect_objects
from solvers.gemini import gemini_solve_math, refine_captcha_guess

st.title("CAPTCHA Solver Demo")
st.write("Upload one or more CAPTCHA images and choose the type to solve.")

captcha_type = st.selectbox(
    "Select CAPTCHA type:",
    ("Text CAPTCHA", "Math CAPTCHA", "Object CAPTCHA")
)

uploaded_files = st.file_uploader(
    "Upload CAPTCHA images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Initialize session state for files if not present
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

# Update session state files to keep FIFO order and avoid duplicates
if uploaded_files:
    # Convert to dict for quick lookup by name+size (approx unique id)
    existing = {(f.name, f.size) for f in st.session_state.uploaded_files_list}
    for f in uploaded_files:
        if (f.name, f.size) not in existing:
            st.session_state.uploaded_files_list.append(f)
else:
    # If user cleared uploader, clear session list as well
    st.session_state.uploaded_files_list = []

def solve_and_get_data(image_path, filename):
    result = None
    difficulty = None

    if captcha_type == "Text CAPTCHA":
        result = solve_text_captcha(image_path)
        if (result.get("error") or (not result.get("refined") and not result.get("text"))):
            st.warning(f"Low confidence or OCR error for {filename} — refining with Gemini...")
        detected_text = result.get('refined') or result.get('text') or "N/A"
        source = result.get('source', 'unknown')
        difficulty = result.get("difficulty")

        table_data = {
            "Detected Text": [detected_text],
            "Source": [source],
            "Difficulty Score": [f"{difficulty}/10" if difficulty is not None else "N/A"]
        }
        return table_data, difficulty, False  # False => not object table

    elif captcha_type == "Math CAPTCHA":
        result = solve_math_captcha(image_path)
        if not result.get("answer") or result.get("error"):
            st.warning(f"Low confidence or OCR error for {filename} — solving with Gemini...")
            expr = result.get("expression", "")
            result = gemini_solve_math(expr)
        expression = result.get('expression', 'N/A')
        answer = result.get('answer', 'N/A')
        source = result.get('source', 'unknown')
        difficulty = result.get("difficulty")

        table_data = {
            "Expression": [expression],
            "Answer": [answer],
            "Source": [source],
            "Difficulty Score": [f"{difficulty}/10" if difficulty is not None else "N/A"]
        }
        return table_data, difficulty, False

    elif captcha_type == "Object CAPTCHA":
        result = detect_objects(image_path)
        objects = []
        if isinstance(result, dict) and isinstance(result.get("objects"), list):
            objects = result.get("objects")
        elif isinstance(result, list):
            objects = result
        elif isinstance(result, str):
            objects = [result]

        table_rows = []
        for obj in objects:
            if isinstance(obj, dict):
                name = obj.get("name", "Unknown")
                confidence = obj.get("confidence", "N/A")
                if isinstance(confidence, float):
                    confidence = f"{confidence:.3f}"
                table_rows.append({"Object": name, "Confidence": confidence})
            else:
                table_rows.append({"Object": str(obj), "Confidence": "N/A"})

        difficulty = result.get("difficulty") if isinstance(result, dict) else None
        return table_rows, difficulty, True  # True => object table


if st.session_state.uploaded_files_list:
    for uploaded_file in st.session_state.uploaded_files_list:
        image = Image.open(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        table_data, difficulty, is_object_table = solve_and_get_data(image_path, uploaded_file.name)

        col_img, col_res = st.columns([1, 1.5])

        with col_img:
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)

        with col_res:
            st.markdown(f"### Results for {uploaded_file.name}")
            if is_object_table:
                if table_data:
                    st.table(table_data)
                else:
                    st.warning(f"No objects detected for {uploaded_file.name}.")
            else:
                st.table(table_data)

            if difficulty is not None:
                st.markdown(f"<p style='font-size:16px; color:#F39C12;'><b>Difficulty Score:</b> {difficulty}/10</p>", unsafe_allow_html=True)
            st.markdown("---")
