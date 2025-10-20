"""
Audio Handler Module for FIND.it
Handles Text-to-Speech (TTS) and Speech-to-Text (STT)
"""

import speech_recognition as sr
import pyttsx3
import threading
import time
from typing import Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class AudioHandler:
    """Handles voice input/output for accessibility"""
    
    def __init__(self, voice_rate: int = 175, voice_volume: float = 0.9):
        """
        Initialize audio handler
        
        Args:
            voice_rate: Speech rate (words per minute, default 175)
            voice_volume: Volume level 0.0-1.0 (default 0.9)
        """
        # Initialize TTS engine
        print("🔊 Initializing Text-to-Speech engine...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', voice_rate)
        self.tts_engine.setProperty('volume', voice_volume)
        
        # Try to set a clear voice (prefer female voices for accessibility)
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        print("✅ TTS engine ready!")
        
        # Initialize STT recognizer
        print("🎤 Initializing Speech-to-Text engine...")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            print("🎤 Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("✅ STT engine ready!")
        
        # Threading for non-blocking TTS
        self.tts_thread = None
        self.is_speaking = False
    
    def speak(self, text: str, blocking: bool = False) -> None:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            blocking: If True, wait until speech completes (default: False)
        """
        if not text or text.strip() == "":
            return
        
        print(f"🔊 Speaking: '{text}'")
        
        if blocking:
            self._speak_sync(text)
        else:
            # Non-blocking mode
            if self.tts_thread and self.tts_thread.is_alive():
                # Wait for current speech to finish
                self.tts_thread.join()
            
            self.tts_thread = threading.Thread(target=self._speak_sync, args=(text,))
            self.tts_thread.start()
    
    def _speak_sync(self, text: str) -> None:
        """Internal synchronous speech method"""
        self.is_speaking = True
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"⚠️  TTS Error: {e}")
        finally:
            self.is_speaking = False
    
    def stop_speaking(self) -> None:
        """Stop current speech immediately"""
        if self.is_speaking:
            self.tts_engine.stop()
            self.is_speaking = False
    
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """
        Listen for voice input from microphone
        
        Args:
            timeout: Seconds to wait for speech to start (default: 5)
            phrase_time_limit: Maximum seconds for phrase (default: 10)
            
        Returns:
            Recognized text or None if failed
        """
        print("🎤 Listening... (speak now)")
        
        try:
            with self.microphone as source:
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            print("🎤 Processing speech...")
            
            # Recognize speech using Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Recognized: '{text}'")
                return text
            except sr.UnknownValueError:
                print("⚠️  Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"⚠️  Speech recognition service error: {e}")
                return None
                
        except sr.WaitTimeoutError:
            print("⚠️  No speech detected (timeout)")
            return None
        except Exception as e:
            print(f"⚠️  Error during listening: {e}")
            return None
    
    def listen_continuous(self, 
                         callback: Callable[[str], None],
                         wake_word: Optional[str] = None,
                         stop_phrase: str = "stop listening") -> None:
        """
        Continuously listen for voice commands
        
        Args:
            callback: Function to call with recognized text
            wake_word: Optional wake word to activate (e.g., "hey assistant")
            stop_phrase: Phrase to stop listening (default: "stop listening")
        """
        print("🎤 Starting continuous listening...")
        if wake_word:
            print(f"   Wake word: '{wake_word}'")
        print(f"   Stop phrase: '{stop_phrase}'")
        
        listening_active = not wake_word  # Active by default if no wake word
        
        with self.microphone as source:
            while True:
                try:
                    print("🎤 Waiting for audio...")
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=10)
                    
                    try:
                        text = self.recognizer.recognize_google(audio)
                        text_lower = text.lower()
                        
                        print(f"🎤 Heard: '{text}'")
                        
                        # Check for stop phrase
                        if stop_phrase.lower() in text_lower:
                            print("🛑 Stop phrase detected!")
                            break
                        
                        # Check for wake word
                        if wake_word and not listening_active:
                            if wake_word.lower() in text_lower:
                                print(f"✅ Wake word detected! Listening active.")
                                listening_active = True
                                self.speak("Yes, how can I help?")
                            continue
                        
                        # Process command
                        if listening_active:
                            callback(text)
                            
                            # If using wake word, deactivate after command
                            if wake_word:
                                listening_active = False
                    
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        print(f"⚠️  Speech recognition error: {e}")
                        time.sleep(1)
                
                except KeyboardInterrupt:
                    print("\n🛑 Stopped by user")
                    break
                except Exception as e:
                    print(f"⚠️  Error: {e}")
                    time.sleep(1)
        
        print("🎤 Continuous listening stopped")
    
    def test_audio(self) -> bool:
        """
        Test both TTS and STT functionality
        
        Returns:
            True if both work, False otherwise
        """
        print("\n" + "=" * 60)
        print("🧪 Testing Audio Handler")
        print("=" * 60)
        
        # Test TTS
        print("\n1️⃣  Testing Text-to-Speech...")
        print("-" * 60)
        test_phrases = [
            "Hello! I am FIND dot it, your visual assistant.",
            "I can help you find objects and read text.",
            "Let's test if you can hear me clearly."
        ]
        
        for phrase in test_phrases:
            self.speak(phrase, blocking=True)
            time.sleep(0.5)
        
        print("✅ TTS test complete!")
        
        # Test STT
        print("\n2️⃣  Testing Speech-to-Text...")
        print("-" * 60)
        self.speak("Please say something after the beep.", blocking=True)
        time.sleep(0.5)
        
        result = self.listen(timeout=5)
        
        if result:
            print("✅ STT test successful!")
            self.speak(f"I heard you say: {result}", blocking=True)
            return True
        else:
            print("❌ STT test failed - no speech detected")
            self.speak("I did not hear anything. Please check your microphone.", blocking=True)
            return False
    
    def emergency_alert(self) -> None:
        """Play emergency alert sound/speech"""
        self.stop_speaking()
        self.tts_engine.setProperty('rate', 200)  # Faster
        self.tts_engine.setProperty('volume', 1.0)  # Max volume
        self.speak("EMERGENCY! EMERGENCY! Calling for help!", blocking=True)
        # Reset to normal
        self.tts_engine.setProperty('rate', 175)
        self.tts_engine.setProperty('volume', 0.9)


# Convenience functions for quick usage
def speak_text(text: str) -> None:
    """Quick TTS function"""
    handler = AudioHandler()
    handler.speak(text, blocking=True)


def listen_once() -> Optional[str]:
    """Quick STT function"""
    handler = AudioHandler()
    return handler.listen()


# Test function
def test_audio_handler():
    """Interactive test for audio handler"""
    print("Testing Audio Handler...")
    print("=" * 60)
    
    try:
        handler = AudioHandler()
        
        # Run full test
        success = handler.test_audio()
        
        if success:
            print("\n✅ All audio tests passed!")
            
            # Interactive test
            print("\n3️⃣  Interactive Test")
            print("-" * 60)
            handler.speak("Let's have a conversation. Say 'exit' to stop.", blocking=True)
            
            while True:
                user_input = handler.listen(timeout=10)
                
                if user_input:
                    if 'exit' in user_input.lower() or 'quit' in user_input.lower():
                        handler.speak("Goodbye!", blocking=True)
                        break
                    
                    # Echo back
                    response = f"You said: {user_input}"
                    handler.speak(response, blocking=True)
                else:
                    break
        
        print("\n" + "=" * 60)
        print("✅ Audio handler test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_audio_handler()