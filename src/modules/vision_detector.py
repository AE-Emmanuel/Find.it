"""
Vision Detection Module for FIND.it
Handles object detection (HuggingFace YOLO) and OCR (pytesseract)
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import YolosImageProcessor, YolosForObjectDetection
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace transformers not available. Install with: pip install transformers torch")


class VisionDetector:
    """Handles object detection and OCR for visual assistance"""
    
    def __init__(self, use_huggingface: bool = True):
        """
        Initialize vision detector
        
        Args:
            use_huggingface: Use HuggingFace YOLO model (True) or fallback to simple detection
        """
        self.use_hf = use_huggingface and HF_AVAILABLE
        self.model = None
        self.processor = None
        
        if self.use_hf:
            try:
                print("Loading HuggingFace YOLO model...")
                # Using YOLOS (YOLO + Transformer) - trained on COCO dataset
                self.processor = YolosImageProcessor.from_pretrained('hustvl/yolos-small')
                self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Failed to load HF model: {e}")
                self.use_hf = False

    def capture_image(self, camera_index: int = 0, save_path: Optional[str] = None, 
                     show_preview: bool = True, preview_duration: int = 3) -> np.ndarray:
        """
        Capture image from camera with optional preview
        
        Args:
            camera_index: Camera device index (0 for default)
            save_path: Optional path to save captured image
            show_preview: Show live preview window (default: True)
            preview_duration: Seconds to show preview before capture (default: 3)
            
        Returns:
            Captured image as numpy array
        """
        import time
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot access camera {camera_index}")
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        print("📸 Camera warming up...")
        
        # Warm up camera
        for i in range(10):
            ret, frame = cap.read()
            time.sleep(0.1)
        
        if show_preview:
            print(f"📸 Camera preview active! Press 'SPACE' to capture or wait {preview_duration}s for auto-capture...")
            print("   Press 'q' to cancel")
            
            start_time = time.time()
            captured_frame = None
            
            while True:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    continue
                
                # Show preview
                display_frame = frame.copy()
                elapsed = int(time.time() - start_time)
                remaining = max(0, preview_duration - elapsed)
                
                # Add countdown text
                cv2.putText(display_frame, f"Press SPACE to capture (auto in {remaining}s)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'q' to cancel", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('FIND.it - Camera Preview', display_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Spacebar
                    print("✅ Manual capture!")
                    captured_frame = frame
                    break
                elif key == ord('q'):  # Quit
                    print("❌ Capture cancelled")
                    cap.release()
                    cv2.destroyAllWindows()
                    raise RuntimeError("Capture cancelled by user")
                elif time.time() - start_time >= preview_duration:  # Auto-capture
                    print("✅ Auto-capture!")
                    captured_frame = frame
                    break
            
            cv2.destroyAllWindows()
            frame = captured_frame
        else:
            # No preview, just capture after warmup
            print("📸 Capturing image...")
            ret, frame = cap.read()
        
        cap.release()
        
        if frame is None:
            raise RuntimeError("Failed to capture image - camera returned empty frame")
        
        # Validate frame is not blank/black
        if frame.mean() < 10:
            raise RuntimeError("Captured frame is too dark - check camera/lighting")
        
        if save_path:
            cv2.imwrite(save_path, frame)
            print(f"💾 Image saved to {save_path}")
        
        return frame
    
    def detect_objects_hf(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects using HuggingFace YOLO model
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of detected objects with metadata
        """
        if not self.use_hf:
            return []
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Run detection
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=confidence_threshold, 
            target_sizes=target_sizes
        )[0]
        
        # Format detections
        detections = []
        h, w = image.shape[:2]
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.tolist()
            x1, y1, x2, y2 = box
            
            # Calculate center and position
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Determine position relative to frame
            position = self._get_spatial_position(center_x, center_y, w, h)
            
            detections.append({
                "object": self.model.config.id2label[label.item()],
                "confidence": score.item(),
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "position": position,
                "distance": "unknown"  # Would need depth sensor for real distance
            })
        
        return detections
    
    def _get_spatial_position(self, x: float, y: float, width: int, height: int) -> str:
        """
        Convert pixel coordinates to spatial description
        
        Args:
            x, y: Object center coordinates
            width, height: Image dimensions
            
        Returns:
            Spatial position description
        """
        # Divide frame into 9 regions (3x3 grid)
        x_third = width / 3
        y_third = height / 3
        
        # Vertical position
        if y < y_third:
            v_pos = "upper"
        elif y < 2 * y_third:
            v_pos = "middle"
        else:
            v_pos = "lower"
        
        # Horizontal position
        if x < x_third:
            h_pos = "left"
        elif x < 2 * x_third:
            h_pos = "center"
        else:
            h_pos = "right"
        
        # Combine for natural language
        if h_pos == "center" and v_pos == "middle":
            return "directly in front of you"
        elif h_pos == "center":
            return f"in the {v_pos} center"
        elif v_pos == "middle":
            return f"on your {h_pos}"
        else:
            return f"in the {v_pos} {h_pos}"
    
    def extract_text_ocr(self, image: np.ndarray, preprocess: bool = True) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image: Input image
            preprocess: Apply preprocessing for better OCR
            
        Returns:
            Extracted text
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if preprocess:
            # Preprocessing for better OCR
            # 1. Increase contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            
            # 2. Denoise
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 3. Threshold
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Run OCR
        try:
            text = pytesseract.image_to_string(gray)
            return text.strip()
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def detect_objects_fallback(self, image: np.ndarray) -> List[Dict]:
        """
        Simple fallback object detection using basic CV techniques
        (Used when HuggingFace not available)
        
        Args:
            image: Input image
            
        Returns:
            List of detected contours/objects
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        h, w = image.shape[:2]
        
        for i, contour in enumerate(contours[:10]):  # Limit to 10 largest
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small objects
                continue
            
            x, y, width, height = cv2.boundingRect(contour)
            center_x = x + width / 2
            center_y = y + height / 2
            
            position = self._get_spatial_position(center_x, center_y, w, h)
            
            detections.append({
                "object": f"object_{i+1}",
                "confidence": 0.7,  # Placeholder
                "bbox": [x, y, x + width, y + height],
                "center": [center_x, center_y],
                "position": position,
                "distance": "unknown"
            })
        
        return detections
    
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Main object detection method (tries HF first, falls back to basic CV)
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detected objects
        """
        if self.use_hf:
            return self.detect_objects_hf(image, confidence_threshold)
        else:
            return self.detect_objects_fallback(image)
    
    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Original image
            detections: List of detected objects
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['object']}: {det['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated


# Test function
def test_vision_detector():
    """Test vision detector with sample image or camera"""
    print("Testing Vision Detector...")
    print("=" * 60)
    
    detector = VisionDetector(use_huggingface=True)
    
    # Test 1: Try to capture from camera
    print("\nTest 1: Camera Capture")
    print("-" * 60)
    try:
        image = detector.capture_image(save_path="test_capture.jpg")
        print(f"✅ Image captured: {image.shape}")
    except Exception as e:
        print(f"⚠️  Camera test skipped: {e}")
        print("Creating synthetic test image instead...")
        # Create a simple test image
        image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), -1)
        cv2.circle(image, (500, 200), 50, (0, 255, 0), -1)
    
    # Test 2: Object detection
    print("\nTest 2: Object Detection")
    print("-" * 60)
    detections = detector.detect_objects(image, confidence_threshold=0.5)
    
    if detections:
        print(f"Detected {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['object']}: {det['confidence']:.2f} @ {det['position']}")
    else:
        print("No objects detected")
    
    # Test 3: OCR
    print("\nTest 3: OCR Text Extraction")
    print("-" * 60)
    # Create test image with text
    text_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(text_image, "FIND.it TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    extracted_text = detector.extract_text_ocr(text_image)
    print(f"Extracted text: '{extracted_text}'")
    
    # Test 4: Annotated output
    print("\nTest 4: Creating Annotated Image")
    print("-" * 60)
    if detections:
        annotated = detector.annotate_image(image, detections)
        cv2.imwrite("test_annotated.jpg", annotated)
        print("✅ Annotated image saved as 'test_annotated.jpg'")
    
    print("\n" + "=" * 60)
    print("✅ All vision tests completed!")


if __name__ == "__main__":
    test_vision_detector()