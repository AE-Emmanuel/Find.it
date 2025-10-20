(# FIND.it — Visual Assistance for the Visually Impaired)

FIND.it is an accessible, prototype visual-assistant application built with Streamlit, OpenCV, HuggingFace object-detection models, OCR (Tesseract), and a conversational ASI client. The app helps users with low or no vision by describing scenes, reading text, locating objects, and providing spoken guidance including an emergency alert feature.

This repository contains a small suite of modules that demonstrate how computer vision + speech interfaces can be combined to build practical assistive tooling.

## Key features

- Capture images from a camera and show annotated results.
- Object detection using a HuggingFace YOLO/transformer model (fallback basic CV if not available).
- OCR text extraction using Tesseract and optional cleanup via ASI.
- Local rule-based intent parsing for voice commands (find object, read text, describe scene, emergency).
- Text-to-speech (pyttsx3) and speech-to-text (SpeechRecognition) support for a voice-first UX.
- ASI client wrapper to generate helpful natural-language descriptions and guidance.

## Repository layout

```
app.py                 # Streamlit application entrypoint
pyproject.toml         # Project metadata
requirements.txt       # Pin dependencies used during development
README.md
src/
	modules/
		asi_client.py      # ASI API wrapper and local intent parser
		vision_detector.py # Object detection, OCR, image utilities
		audio_handler.py   # TTS and STT utilities
```

## Quick start (developer / local)

Prerequisites

- Python 3.11+ (pyproject suggests 3.13 but most code runs on 3.11/3.12; adjust if needed)
- Tesseract OCR installed on your system (pytesseract is a Python wrapper)
- A working microphone and (optional) webcam for interactive features

Install OS-level Tesseract (macOS example with Homebrew):

```bash
# macOS (Homebrew)
brew install tesseract
```

Create and activate a virtual environment, then install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables

Create a `.env` file in the project root and add your ASI API key (used by `ASIClient`):

```
ASI_API_KEY=your_api_key_here
# Optional: ASI_SESSION_ID if you want to pin a session id
```

Run the Streamlit app

```bash
streamlit run app.py
```

This will open a browser UI where you can capture images, select modes (Find Object, Read Text, Describe Scene, Voice Command, Emergency), and interact using speech.

Notes and troubleshooting

- HuggingFace YOLO model: The repo attempts to use a YOLO/transformer model from `transformers` and `torch`. If these packages are not available or you have limited GPU/CPU resources, the code falls back to a simple computer-vision contour detector in `vision_detector.py`.
- Microphone permissions: On macOS you must allow Terminal (or your IDE) access to the microphone in System Settings.
- Tesseract: If pytesseract raises errors, ensure Tesseract is on your PATH. You can test it with `tesseract --version`.
- PyAudio: Installing `pyaudio` may require portaudio headers. On macOS install via Homebrew: `brew install portaudio` and then `pip install pyaudio`.

## Modules overview

- `app.py` — Streamlit front-end. Initializes `ASIClient`, `VisionDetector`, and `AudioHandler` and wires UI flows:
	- Capture image
	- Find object workflow (detect -> ASI guidance -> speak/annotate)
	- Read text workflow (OCR -> ASI cleaning -> speak)
	- Describe scene workflow (detect -> ASI description -> speak/annotate)
	- Emergency mode: raises audio alert and displays prominent UI

- `src/modules/asi_client.py` — Handles ASI API requests with caching, a local rule-based intent parser (no network calls required for intent detection), and helper methods to: `describe_scene`, `find_object_guidance`, and `clean_ocr_text`.

- `src/modules/vision_detector.py` — Handles image capture, object detection using HuggingFace YOLO (if available), a fallback contour-based detector, OCR via pytesseract, and image annotation utilities.

- `src/modules/audio_handler.py` — Wraps `pyttsx3` for TTS and `SpeechRecognition` for STT. Also provides convenience helpers and emergency alert behavior.

## Testing

Each module contains a small `test_*` function that can be run directly for local, manual testing. For example:

```bash
python src/modules/asi_client.py
python src/modules/vision_detector.py
python src/modules/audio_handler.py
```

These tests are not formal unit tests but quick smoke checks to validate environment, dependencies, and hardware.

## Development notes and suggestions

- Model selection: The code uses a small YOLOS model (`hustvl/yolos-small`) through HuggingFace. If you intend to support more categories or higher accuracy, consider switching to a larger YOLOv8/YOLOv7 model or a custom model trained on target items.
- Performance: Running the HF model on CPU can be slow; using a GPU with matching PyTorch installation will speed up detections considerably.
- Packaging: If you plan to publish, update `pyproject.toml` metadata and Python version constraints.
- Security: Keep `ASI_API_KEY` secret. Do not commit `.env` with API keys.

## Contributing

Contributions are welcome. Suggested small tasks:

- Add unit tests for intent parsing and OCR cleaning functions.
- Add a Dockerfile for consistent runtime environment.
- Improve the object label mapping / custom vocabulary for commonly searched items.

If you open issues or pull requests, include a short description, the platform used (macOS/Linux/Windows), Python version, and steps to reproduce.

## License

This project does not include a license file. Add a LICENSE (e.g., MIT) if you want to make the repository open-source.

## Acknowledgements

- Built with Streamlit, OpenCV, HuggingFace transformers, Tesseract OCR, and SpeechRecognition.

---

If you'd like, I can also:

- Add a short GIF or screenshots to the README showing the Streamlit UI.
- Create a CONTRIBUTING.md and ISSUE_TEMPLATE.md.
- Add a minimal GitHub Actions workflow to run linting and smoke tests on pushes.

Tell me which extras you want and I'll add them.
