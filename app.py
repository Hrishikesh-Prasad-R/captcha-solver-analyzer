"""
CAPTCHA Solver Dashboard - Clean Professional UI
Industrial-grade solving with confidence classification and explainability.
"""

import os
import logging
import tempfile
import time
from typing import Dict, Any

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Initialize determinism FIRST
from utils.determinism import set_seed, get_deterministic_hash
from utils.jitter import calculate_jitter
from app_config import settings
from models.confidence_state import ConfidenceState, StateThresholds

# Set global seed
set_seed(settings.random_seed)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import solvers
from solvers.text_captcha import solve_text_captcha
from solvers.math_captcha import solve_math_captcha
from solvers.object_captcha import detect_objects

# Import classifier for auto-detection
from utils.classifier import classify_captcha, CaptchaType

# --- Page Config ---
st.set_page_config(
    page_title="CAPTCHA Solver",
    page_icon="🔓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Clean Professional CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Answer Display */
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        letter-spacing: 3px;
        text-align: center;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* State Badges */
    .state-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 13px;
        margin: 0.5rem 0;
    }
    
    .state-safe {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #22c55e;
    }
    
    .state-review {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
    }
    
    .state-noaction {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #ef4444;
    }
    
    /* Metric Boxes */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-box {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Phase Timeline */
    .phase-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        background: #f9fafb;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .phase-header {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    
    .phase-action {
        font-size: 0.875rem;
        color: #6b7280;
        font-style: italic;
    }
    
    .phase-details {
        font-size: 0.8rem;
        color: #4b5563;
        margin-top: 0.5rem;
        padding-left: 1rem;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #9ca3af;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* File name badge */
    .file-badge {
        display: inline-block;
        background: #e5e7eb;
        color: #374151;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Color Palette for Object Detection ---
COLOR_PALETTE = [
    (255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0),
    (138, 43, 226), (255, 105, 180), (0, 206, 209), (255, 140, 0),
]


def get_color_for_label(label: str, label_colors: Dict[str, tuple]) -> tuple:
    if label not in label_colors:
        color_index = get_deterministic_hash(label) % len(COLOR_PALETTE)
        label_colors[label] = COLOR_PALETTE[color_index]
    return label_colors[label]


def get_solver_for_type(ctype: str):
    if ctype == "Text CAPTCHA":
        return solve_text_captcha
    elif ctype == "Math CAPTCHA":
        return solve_math_captcha
    else:
        return detect_objects


def solve_and_get_data(image_path: str, image: Image.Image, use_fallback: bool, captcha_type: str, enable_jitter: bool, target_filter: str = "") -> Dict[str, Any]:
    start_time = time.time()
    solver = get_solver_for_type(captcha_type)
    
    if enable_jitter:
        jitter_score, outputs, _ = calculate_jitter(
            solver, image_path, num_runs=StateThresholds.JITTER_RUNS,
            use_gemini_fallback=use_fallback
        )
        result = solver(image_path, use_gemini_fallback=use_fallback)
        result["jitter_score"] = jitter_score
    else:
        result = solver(image_path, use_gemini_fallback=use_fallback)
        result["jitter_score"] = 0.0
    
    elapsed_time = time.time() - start_time
    
    # Process object detection
    if captcha_type == "Object CAPTCHA":
        objects = result.get("objects", [])
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()
        label_colors: Dict[str, tuple] = {}
        
        # Filter objects if target_filter is provided
        if target_filter:
            filtered_objects = [
                obj for obj in objects 
                if target_filter in obj.get("name", "").lower()
            ]
            result["filter_applied"] = target_filter
            result["filter_match_count"] = len(filtered_objects)
            result["all_objects"] = objects  # Keep original for debug
            objects_to_draw = filtered_objects
        else:
            result["filter_applied"] = None
            objects_to_draw = objects
        
        # Draw bounding boxes for filtered objects only
        for obj in objects_to_draw:
            name = obj.get("name", "Unknown")
            bbox = obj.get("bbox")
            color = get_color_for_label(name, label_colors)
            
            if bbox:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, max(0, y1 - 14)), name, fill=color, font=font)
        
        result["annotated_image"] = annotated_image
        result["displayed_objects"] = objects_to_draw
    else:
        result["annotated_image"] = image
    
    result["elapsed_time"] = elapsed_time
    return result


# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🔓 CAPTCHA Solver</h1>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    
    captcha_type_selection = st.selectbox(
        "CAPTCHA Type",
        ("Text CAPTCHA", "Math CAPTCHA", "Object CAPTCHA"),
        help="Select the type of CAPTCHA to solve"
    )
    
    use_ai_fallback = st.toggle(
        "🤖 Gemini AI Fallback",
        value=settings.use_gemini_fallback,
        help="Use Gemini to refine OCR results"
    )
    
    enable_jitter_check = st.toggle(
        "📊 Jitter Detection",
        value=False,
        help=f"Run {StateThresholds.JITTER_RUNS}x for consistency check"
    )
    
    # Target object filter (only for Object CAPTCHA)
    target_object_filter = ""
    if captcha_type_selection == "Object CAPTCHA":
        st.divider()
        target_object_filter = st.text_input(
            "🎯 Target Object",
            placeholder="e.g., car, person, dog",
            help="Leave empty to show all objects, or type an object name to filter"
        ).strip().lower()
    
    st.divider()
    
    # Sample Images Section
    st.subheader("📁 Sample Images")
    
    # Map CAPTCHA type to sample folder
    sample_folder_map = {
        "Text CAPTCHA": "text_captcha",
        "Math CAPTCHA": "math_captcha",
        "Object CAPTCHA": "object_captcha",
    }
    sample_folder = os.path.join("samples", sample_folder_map[captcha_type_selection])
    
    # Get available samples
    sample_files = []
    if os.path.exists(sample_folder):
        sample_files = sorted([f for f in os.listdir(sample_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    selected_sample = st.selectbox(
        "Use a sample image",
        options=["None (upload your own)"] + sample_files,
        help="Select a sample image to test without uploading"
    )
    
    st.divider()
    st.caption("Deterministic • Type-Safe")

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "📤 Upload CAPTCHA Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload one or more CAPTCHA images to solve"
)

# --- Build list of images to process ---
images_to_process = []

# Priority 1: Uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        images_to_process.append({
            "source": "upload",
            "name": uploaded_file.name,
            "image": Image.open(uploaded_file)
        })
# Priority 2: Selected sample (if no uploads)
elif selected_sample != "None (upload your own)":
    sample_path = os.path.join(sample_folder, selected_sample)
    if os.path.exists(sample_path):
        images_to_process.append({
            "source": "sample",
            "name": f"📁 {selected_sample}",
            "image": Image.open(sample_path),
            "path": sample_path  # Can use directly, no temp file needed
        })

# --- Process Results ---
if images_to_process:
    for idx, img_data in enumerate(images_to_process):
        image = img_data["image"]
        image_name = img_data["name"]
        
        # For sample images, use existing path; for uploads, create temp file
        if img_data["source"] == "sample":
            image_path = img_data["path"]
            needs_cleanup = False
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image.save(tmp.name)
                image_path = tmp.name
                needs_cleanup = True

        try:
            # Determine CAPTCHA type (now always manual selection)
            captcha_type = captcha_type_selection
            
            with st.spinner(f"🔍 Solving {captcha_type}..."):
                result = solve_and_get_data(image_path, image, use_ai_fallback, captcha_type, enable_jitter_check, target_object_filter)
            
            # Add result metadata
            result["captcha_type"] = captcha_type
            
            # Result Container
            st.divider()
            
            # Two columns: Image | Results
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.image(result.get("annotated_image", image), use_container_width=True)
                st.markdown(f'<span class="file-badge">{image_name}</span>', unsafe_allow_html=True)
            
            with col2:
                # State Badge
                state = result.get("state", "NO_ACTION")
                state_reason = result.get("state_reason", "")
                
                state_config = {
                    "SAFE_OUTPUT": ("✅", "state-safe", "SAFE OUTPUT"),
                    "REVIEW_REQUIRED": ("⚠️", "state-review", "NEEDS REVIEW"),
                    "NO_ACTION": ("🚫", "state-noaction", "NO ACTION"),
                }
                emoji, css_class, label = state_config.get(state, ("❓", "state-noaction", "UNKNOWN"))
                
                st.markdown(f'''
                    <div class="state-badge {css_class}">
                        <span>{emoji}</span>
                        <span>{label}</span>
                    </div>
                    <p style="color: #6b7280; font-size: 0.8rem; margin-top: 4px;">{state_reason}</p>
                ''', unsafe_allow_html=True)
                
                # Main Result - The Answer
                if captcha_type == "Text CAPTCHA":
                    text = result.get("text") or result.get("refined") or "—"
                    st.markdown(f'<div class="answer-box">{text}</div>', unsafe_allow_html=True)
                    
                elif captcha_type == "Math CAPTCHA":
                    expr = result.get("expression") or "—"
                    answer = result.get("answer") or "—"
                    st.markdown(f'<div class="answer-box">{expr} = {answer}</div>', unsafe_allow_html=True)
                    
                else:  # Object CAPTCHA
                    # Use displayed_objects if filter was applied, otherwise use all objects
                    displayed_objs = result.get("displayed_objects", result.get("objects", []))
                    filter_applied = result.get("filter_applied")
                    
                    if filter_applied and len(displayed_objs) == 0:
                        # Filter was applied but no matches found
                        st.markdown(f'<div class="answer-box" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">🚫 Object not found: "{filter_applied}"</div>', unsafe_allow_html=True)
                    else:
                        obj_names = [o.get("name", "?") for o in displayed_objs[:5]]
                        st.markdown(f'<div class="answer-box">{", ".join(obj_names) or "No objects"}</div>', unsafe_allow_html=True)
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    conf = result.get("confidence")
                    conf_str = f"{conf:.0%}" if conf else "—"
                    st.metric("Confidence", conf_str)
                
                with m2:
                    diff = result.get("difficulty")
                    diff_str = f"{diff}/10" if diff else "—"
                    st.metric("Difficulty", diff_str)
                
                with m3:
                    time_str = f"{result.get('elapsed_time', 0):.2f}s"
                    st.metric("Time", time_str)
                
                # Jitter
                if enable_jitter_check:
                    jitter = result.get("jitter_score", 0)
                    jitter_color = "green" if jitter == 0 else "orange" if jitter <= 0.3 else "red"
                    st.progress(min(jitter, 1.0), text=f"Jitter: {jitter:.0%}")
            
            # Expandable Details Section
            phases = result.get("phases", [])
            if phases:
                with st.expander("📋 View Processing Details", expanded=False):
                    for phase in phases:
                        step = phase.get("step", "?")
                        phase_name = phase.get("phase", "Unknown")
                        action = phase.get("action", "")
                        status = phase.get("status", "")
                        
                        status_emoji = {
                            "success": "✅", "failed": "❌", "warning": "⚠️",
                            "complete": "🏁", "fallback": "🔄", "attempting": "⏳",
                            "no_detection": "🔍", "fallback_failed": "⚠️"
                        }.get(status, "•")
                        
                        st.markdown(f'''
                            <div class="phase-item">
                                <div class="phase-header">{status_emoji} Step {step}: {phase_name}</div>
                                <div class="phase-action">{action}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        # Show key details
                        if "calculation" in phase:
                            st.code(phase["calculation"], language=None)
                        
                        if "raw_detections" in phase:
                            for det in phase["raw_detections"]:
                                st.text(f"  → '{det['text']}' ({det['confidence']})")
            
            # Debug Data (always available as expander)
            with st.expander("🔧 View Debug Data", expanded=False):
                debug_data = {k: v for k, v in result.items() if k not in ["annotated_image", "phases"]}
                st.json(debug_data)
        
        finally:
            if needs_cleanup:
                try:
                    os.unlink(image_path)
                except:
                    pass

else:
    # Empty State
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📤</div>
            <h3>Upload CAPTCHA Images</h3>
            <p>Drop your images above to get started</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">Supports PNG, JPG, JPEG</p>
        </div>
    """, unsafe_allow_html=True)