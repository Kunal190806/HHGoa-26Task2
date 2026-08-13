import os
import requests
import tempfile
import base64
from typing import Optional

class SarvamClient:
    def __init__(self):
        self.api_key = os.environ.get("SARVAM_API_KEY", "sk_fpcbmwwl_6VOhW9y2L2BbcrhB5uzvlRST")
        self.base_url = "https://api.sarvam.ai"

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Sends audio to Sarvam STT API and returns transcribed text.
        Falls back to a mock response only if no API key is set at all.
        """
        if not self.api_key:
            print("WARNING: No SARVAM_API_KEY found. Mocking STT response.")
            return "This is a mocked transcription because the Sarvam API key is missing. What is the capital of India?"
            
        # Write bytes to a temporary file since requests likes file objects for multipart
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            url = f"{self.base_url}/speech-to-text" # Using the base STT endpoint
            headers = {
                "api-subscription-key": self.api_key
            }
            
            # Using multipart/form-data
            with open(tmp_path, "rb") as f:
                files = {"file": (filename, f, "audio/wav")}
                # Depending on exact Sarvam API requirements, we might need model or language
                data = {"model": "saarika:v2.5"} 
                
                response = requests.post(url, headers=headers, files=files, data=data)
                
            if response.status_code == 200:
                result = response.json()
                return result.get("transcript", "")
            else:
                print(f"Sarvam STT Error: {response.status_code} - {response.text}")
                # Graceful degradation
                return "Could not transcribe audio due to API error."
                
        except Exception as e:
            print(f"Exception calling Sarvam STT: {e}")
            return "Could not transcribe audio due to internal error."
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
