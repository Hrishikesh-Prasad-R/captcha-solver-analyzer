# CAPTCHA Solver Analyzer

![CAPTCHA Solver Banner](https://img.shields.io/badge/Streamlit-CAPTCHA_Solver-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Industrial Grade](https://img.shields.io/badge/Quality-Industrial_Grade-gold)

A powerful and versatile **CAPTCHA solving application** built with Streamlit, designed to automatically detect and solve multiple types of CAPTCHA images.

## 🏭 Industrial Grade Features

- **Deterministic Processing**: Fixed random seeds ensure reproducible results
- **Type Safety**: Pydantic models for all data structures
- **Lazy Loading**: Models load on-demand to prevent startup crashes
- **Structured Logging**: Machine-readable logs for production monitoring
- **Centralized Config**: Environment-based configuration with validation

---

## 🚀 Features

- **Multi-type CAPTCHA support:**

  - **Text CAPTCHA** — OCR-based text extraction with Gemini AI refinement
  - **Math CAPTCHA** — Automatic math expression recognition and solving
  - **Object CAPTCHA** — YOLOv8-powered object detection with bounding boxes

- **AI Fallback**: Optional Gemini 1.5 Flash integration (Free Tier compatible)
- **Batch Upload**: Process multiple images at once
- **Debug Mode**: Toggle detailed output for troubleshooting

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Hrishikesh-Prasad-R/captcha-solver-analyzer.git
cd captcha-solver-analyzer
```

### 2. Create and activate virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
# Create .env file
echo "API_KEY=your_gemini_api_key_here" > .env
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t captcha-solver .

# Run the container
docker run -p 8501:8501 -e API_KEY=your_key captcha-solver
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=solvers --cov=models

# Run type checking
mypy .

# Run linter
ruff check .
```

---

## 📁 Project Structure

```
captcha-solver-analyzer/
├── app.py                 # Streamlit main application
├── config.py              # Centralized configuration (pydantic-settings)
├── Dockerfile             # Production container
├── pyproject.toml         # Tool configurations (ruff, pytest, mypy)
├── requirements.txt       # Python dependencies
├── models/
│   └── captcha_result.py  # Pydantic result models
├── solvers/
│   ├── gemini.py          # Gemini AI integration
│   ├── math_captcha.py    # Math CAPTCHA solver
│   ├── object_captcha.py  # YOLOv8 object detection
│   ├── ocr_reader.py      # EasyOCR reader singleton
│   └── text_captcha.py    # Text CAPTCHA solver
├── tests/
│   └── test_solvers.py    # Unit tests
└── utils/
    └── determinism.py     # Seed management for reproducibility
```

---

## ⚙️ Configuration

All configuration is managed via environment variables or `.env` file:

| Variable                     | Default | Description                       |
| ---------------------------- | ------- | --------------------------------- |
| `API_KEY`                    | None    | Google Gemini API key             |
| `RANDOM_SEED`                | 42      | Global seed for reproducibility   |
| `OCR_CONFIDENCE_THRESHOLD`   | 0.5     | Minimum OCR confidence            |
| `OBJECT_DETECTION_THRESHOLD` | 0.5     | Minimum YOLO detection confidence |
| `USE_GEMINI_FALLBACK`        | true    | Enable AI fallback                |

---

## 🔒 About the Pretrained Model

This project uses **YOLOv8n** (Nano), a pretrained object detection model from [Ultralytics](https://github.com/ultralytics/ultralytics).

- **No training required**: The model is automatically downloaded on first run
- **COCO dataset**: Pretrained on 80 common object classes
- **Deterministic inference**: Results are reproducible with fixed seeds

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
