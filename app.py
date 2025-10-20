"""
FIND.it - Visual Assistance App for Blind Users
Main Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'modules'))

from asi_client import ASIClient
from vision_detector import VisionDetector
from audio_handler import AudioHandler


# Page configuration
st.set_page_config(
    page_title="FIND.it - Visual Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better accessibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    .success-box {
        background-color: #C8E6C9;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        background-color: #FFF9C4;
        border-left: 5px solid #FFC107;
    }
    .error-box {
        background-color: #FFCDD2;
        border-left: 5px solid #F44336;
    }
    .emergency-box {
        background-color: #FF0000;
        color: white;
        padding: 2rem;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        border-radius: 1rem;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .stButton>button {
        font-size: 1.5rem;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'asi_client' not in st.session_state:
    st.session_state.asi_client = None
if 'vision_detector' not in st.session_state:
    st.session_state.vision_detector = None
if 'audio_handler' not in st.session_state:
    st.session_state.audio_handler = None
if 'last_image' not in st.session_state:
    st.session_state.last_image = None
if 'last_detections' not in st.session_state:
    st.session_state.last_detections = []
if 'emergency_mode' not in st.session_state:
    st.session_state.emergency_mode = False


@st.cache_resource
def initialize_systems():
    """Initialize all systems (cached to avoid reloading)"""
    with st.spinner("🔧 Initializing FIND.it systems..."):
        asi = ASIClient()
        vision = VisionDetector(use_huggingface=True)
        audio = AudioHandler()
        return asi, vision, audio


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">👁️ FIND.it</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Visual Assistance for the Visually Impaired</p>', unsafe_allow_html=True)
    
    # Initialize systems
    if st.session_state.asi_client is None:
        try:
            asi, vision, audio = initialize_systems()
            st.session_state.asi_client = asi
            st.session_state.vision_detector = vision
            st.session_state.audio_handler = audio
            st.success("✅ All systems initialized!")
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            return
    
    # Emergency mode display
    if st.session_state.emergency_mode:
        st.markdown('<div class="emergency-box">🚨 EMERGENCY MODE 🚨<br>CALLING 911</div>', unsafe_allow_html=True)
        if st.button("Exit Emergency Mode", key="exit_emergency"):
            st.session_state.emergency_mode = False
            st.rerun()
        return
    
    # Sidebar - Controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        mode = st.radio(
            "Select Mode:",
            ["🎤 Voice Command", "📸 Find Object", "📖 Read Text", "👁️ Describe Scene", "🚨 Emergency"],
            key="mode_selector"
        )
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider("Detection Confidence", 0.3, 0.9, 0.5, 0.05)
        voice_rate = st.slider("Voice Speed (WPM)", 100, 250, 175, 25)
        
        if st.button("🔧 Update Settings"):
            st.session_state.audio_handler.tts_engine.setProperty('rate', voice_rate)
            st.success("Settings updated!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Camera Feed")
        camera_placeholder = st.empty()
        
        if st.button("📸 Capture Image", key="capture_btn", use_container_width=True):
            with st.spinner("📸 Capturing image..."):
                try:
                    image = st.session_state.vision_detector.capture_image(
                        save_path="temp_capture.jpg",
                        show_preview=False,
                        preview_duration=0
                    )
                    st.session_state.last_image = image
                    camera_placeholder.image(image, channels="BGR", caption="Captured Image")
                    st.success("✅ Image captured!")
                except Exception as e:
                    st.error(f"❌ Capture failed: {e}")
        
        # Show last captured image
        if st.session_state.last_image is not None:
            camera_placeholder.image(st.session_state.last_image, channels="BGR", caption="Last Captured Image")
    
    with col2:
        st.subheader("🤖 Assistant Response")
        response_placeholder = st.empty()
        
        # Mode-specific interface
        if mode == "🎤 Voice Command":
            st.info("🎤 Click 'Listen' and speak your command")
            
            if st.button("🎤 Listen for Command", key="listen_btn", use_container_width=True):
                st.session_state.audio_handler.speak("Listening for your command", blocking=False)
                
                with st.spinner("🎤 Listening..."):
                    command = st.session_state.audio_handler.listen(timeout=5)
                
                if command:
                    st.write(f"**You said:** {command}")
                    
                    # Parse intent with ASI
                    with st.spinner("🤖 Understanding your request..."):
                        intent_data = st.session_state.asi_client.parse_intent(command)
                    
                    st.write(f"**Intent:** {intent_data.get('intent', 'unknown')}")
                    st.write(f"**Confidence:** {intent_data.get('confidence', 0):.0%}")
                    
                    # Execute based on intent
                    execute_intent(intent_data, command, response_placeholder)
                else:
                    st.warning("⚠️ No command detected")
                    st.session_state.audio_handler.speak("I didn't hear anything. Please try again.", blocking=False)
        
        elif mode == "📸 Find Object":
            st.info("🔍 Enter object name and capture image")
            
            target_object = st.text_input("What are you looking for?", placeholder="e.g., keys, phone, cup")
            
            if st.button("🔍 Find Object", key="find_btn", use_container_width=True) and target_object:
                if st.session_state.last_image is not None:
                    find_object_workflow(target_object, response_placeholder, confidence_threshold)
                else:
                    st.warning("⚠️ Please capture an image first!")
        
        elif mode == "📖 Read Text":
            st.info("📖 Capture image containing text")
            
            if st.button("📖 Read Text", key="read_btn", use_container_width=True):
                if st.session_state.last_image is not None:
                    read_text_workflow(response_placeholder)
                else:
                    st.warning("⚠️ Please capture an image first!")
        
        elif mode == "👁️ Describe Scene":
            st.info("👁️ Capture image for scene description")
            
            if st.button("👁️ Describe Scene", key="describe_btn", use_container_width=True):
                if st.session_state.last_image is not None:
                    describe_scene_workflow(response_placeholder, confidence_threshold)
                else:
                    st.warning("⚠️ Please capture an image first!")
        
        elif mode == "🚨 Emergency":
            st.error("🚨 Emergency Mode - Press button to activate")
            
            if st.button("🚨 ACTIVATE EMERGENCY", key="emergency_btn", use_container_width=True):
                st.session_state.emergency_mode = True
                st.session_state.audio_handler.emergency_alert()
                st.rerun()


def execute_intent(intent_data, command, placeholder):
    """Execute action based on parsed intent"""
    intent = intent_data.get('intent', 'describe_scene')
    
    if intent == 'find_object':
        entities = intent_data.get('entities', [])
        if entities:
            target = entities[0]
            placeholder.info(f"🔍 Searching for: {target}")
            find_object_workflow(target, placeholder)
        else:
            placeholder.warning("⚠️ No object specified")
    
    elif intent == 'read_text':
        placeholder.info("📖 Reading text from image...")
        read_text_workflow(placeholder)
    
    elif intent == 'describe_scene':
        placeholder.info("👁️ Analyzing scene...")
        describe_scene_workflow(placeholder)
    
    elif intent == 'emergency':
        st.session_state.emergency_mode = True
        st.session_state.audio_handler.emergency_alert()
        st.rerun()
    
    else:
        placeholder.info("🤖 Processing your request...")
        response = f"I understood: {command}"
        placeholder.write(response)
        st.session_state.audio_handler.speak(response, blocking=False)


def find_object_workflow(target_object, placeholder, confidence_threshold=0.5):
    """Find specific object workflow"""
    with st.spinner("🔍 Detecting objects..."):
        detections = st.session_state.vision_detector.detect_objects(
            st.session_state.last_image,
            confidence_threshold=confidence_threshold
        )
        st.session_state.last_detections = detections
    
    if detections:
        # Use ASI to generate guidance
        guidance = st.session_state.asi_client.find_object_guidance(target_object, detections)
        
        placeholder.markdown(f'<div class="status-box success-box">🎯 {guidance}</div>', unsafe_allow_html=True)
        st.session_state.audio_handler.speak(guidance, blocking=False)
        
        # Show annotated image
        annotated = st.session_state.vision_detector.annotate_image(
            st.session_state.last_image,
            detections
        )
        st.image(annotated, channels="BGR", caption="Detected Objects")
    else:
        msg = "No objects detected in the image."
        placeholder.warning(msg)
        st.session_state.audio_handler.speak(msg, blocking=False)


def read_text_workflow(placeholder):
    """Read text from image workflow"""
    with st.spinner("📖 Extracting text..."):
        raw_text = st.session_state.vision_detector.extract_text_ocr(
            st.session_state.last_image
        )
    
    if raw_text and raw_text.strip():
        # Clean text with ASI
        with st.spinner("🤖 Cleaning text..."):
            cleaned_text = st.session_state.asi_client.clean_ocr_text(raw_text)
        
        placeholder.markdown(f'<div class="status-box success-box">📖 Text Found:<br><br>{cleaned_text}</div>', unsafe_allow_html=True)
        st.session_state.audio_handler.speak(cleaned_text, blocking=False)
        
        with st.expander("Show Raw OCR Output"):
            st.code(raw_text)
    else:
        msg = "No text detected in the image."
        placeholder.warning(msg)
        st.session_state.audio_handler.speak(msg, blocking=False)


def describe_scene_workflow(placeholder, confidence_threshold=0.5):
    """Describe scene workflow"""
    with st.spinner("👁️ Analyzing scene..."):
        detections = st.session_state.vision_detector.detect_objects(
            st.session_state.last_image,
            confidence_threshold=confidence_threshold
        )
        st.session_state.last_detections = detections
    
    if detections:
        # Extract object names and locations
        object_names = [d['object'] for d in detections]
        object_locations = [
            {
                'object': d['object'],
                'position': d['position'],
                'confidence': d['confidence']
            }
            for d in detections
        ]
        
        # Generate description with ASI
        description = st.session_state.asi_client.describe_scene(
            object_names,
            object_locations
        )
        
        placeholder.markdown(f'<div class="status-box success-box">👁️ Scene Description:<br><br>{description}</div>', unsafe_allow_html=True)
        st.session_state.audio_handler.speak(description, blocking=False)
        
        # Show annotated image
        annotated = st.session_state.vision_detector.annotate_image(
            st.session_state.last_image,
            detections
        )
        st.image(annotated, channels="BGR", caption="Scene Analysis")
        
        # Show detection details
        with st.expander("Show Detection Details"):
            for det in detections:
                st.write(f"**{det['object']}** - {det['confidence']:.0%} confidence @ {det['position']}")
    else:
        msg = "No objects detected in the scene."
        placeholder.warning(msg)
        st.session_state.audio_handler.speak(msg, blocking=False)


if __name__ == "__main__":
    main()