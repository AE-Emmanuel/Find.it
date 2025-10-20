# FIND.it — Visual Assistance for the Visually Impaired

FIND.it is an accessible, prototype visual-assistant application built with Streamlit, OpenCV, HuggingFace object-detection models(YOLO v8), OCR (Tesseract), and a conversational ASI client. The app helps users with low or no vision by describing scenes, reading text, locating objects, and providing spoken guidance including an emergency alert feature.

This repository contains a small suite of modules that demonstrate how computer vision + speech interfaces can be combined to build practical assistive tooling.

## Key features

- Capture images from a camera and show annotated results.
- Object detection using a HuggingFace YOLO/transformer model.
- OCR text extraction using Tesseract and optional cleanup via ASI.
- Local rule-based intent parsing for voice commands (find object, read text, describe scene, emergency).
- Text-to-speech (pyttsx3) and speech-to-text (SpeechRecognition).
- ASI client wrapper to generate helpful natural-language descriptions and guidance.


Prerequisites

- Python 3.11+ 
- Tesseract OCR installed on your system 
- A working microphone 

Install OS-level Tesseract 

```bash
# macOS (Homebrew)
brew install tesseract
```


## Development notes and suggestions
- Model selection: The code uses a small YOLOS model (`hustvl/yolos-small`) through HuggingFace. If you intend to support more categories or higher accuracy, consider switching to a larger YOLOv8/YOLOv7 model or a custom model trained on target items.
- Performance: Running the HF model on CPU can be slow; using a GPU with matching PyTorch installation will speed up detections considerably.



