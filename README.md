# Hrishikesh-Prasad-R CAPTCHA Solver Analyzer

![CAPTCHA Solver Banner](https://img.shields.io/badge/Streamlit-CAPTCHA_Solver-blue)

A powerful and versatile **CAPTCHA solving application** built with Streamlit, designed to automatically detect and solve multiple types of CAPTCHA images with ease. 

---

## 🚀 Features

- **Multi-type CAPTCHA support:**
  - **Text CAPTCHA** — OCR based text extraction with smart refinement using Gemini AI.
  - **Math CAPTCHA** — Automatic math expression recognition and solving.
  - **Object CAPTCHA** — Detect and identify objects in images with confidence scores.

- **Batch Upload:** Upload multiple CAPTCHA images at once for bulk processing.

- **Instant Solving:** Results show immediately upon upload — no button clicks needed.

- **Clean UI:** Side-by-side display of uploaded images and corresponding solution tables for easy analysis.

- **Modular Design:** Separate solver modules under `solvers/` for easy extension and maintenance.

---

## ⚡ Quick Start

1. **Clone the repo:**

```bash
git clone https://github.com/Hrishikesh-Prasad-R/captcha-solver-analyzer.git
cd captcha-solver-analyzer
```

2. **Create and activate a virtual environment:**

```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Windows (CMD)
python -m venv venv
venv\Scripts\activate.bat

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app:**

```bash
streamlit run app.py
```

## Usage

- Open your browser if it doesn’t open automatically.
- Use the sidebar or drag-and-drop area to upload one or more CAPTCHA images.
- The app will automatically detect the CAPTCHA type and solve it instantly.
- View the uploaded images side-by-side with their solutions in a clean, easy-to-read table.
- Explore different CAPTCHA types: Text, Math, and Object recognition.
