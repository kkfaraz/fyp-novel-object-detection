import os
import requests
import time
from typing import List


class TextRefiner:
    """
    Text refiner using local Ollama LLM.
    Robust implementation - handles all errors gracefully.
    """
    
    def __init__(self, device, model: str = "llama3", ollama_url: str = None):
        print(f"Initializing TextRefiner (Local Ollama) on {device}")
        
        self.model = model
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.base_url = f"{self.ollama_url}/api/generate"
        self.is_available = False
        self.error_count = 0
        self.max_errors = 5  # Disable after too many errors
        
        self.min_delay = 0.05
        self.last_request_time = 0

        self.system_prompt = (
            "You are an expert visual analyst. "
            "Refine image descriptions by identifying ALL visible objects at different scales. "
            "Describe observable visual attributes (color, size, shape, material, texture). "
            "Do NOT hallucinate objects not visually present. "
            "Respond with a single continuous description."
        )

        # Test connection and model availability
        try:
            test_resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if test_resp.status_code == 200:
                models = [m["name"] for m in test_resp.json().get("models", [])]
                print(f"[Ollama] Connected to {self.ollama_url}")
                print(f"[Ollama] Available models: {models}")
                
                model_names = [m.split(':')[0] for m in models]
                if self.model in models or self.model.split(':')[0] in model_names:
                    self.is_available = True
                    print(f"[Ollama] Using model: {self.model} ✓")
                else:
                    print(f"[Ollama] Model '{self.model}' not found!")
                    print(f"[Ollama] Run: ollama pull {self.model}")
                    self.is_available = False
            else:
                print(f"[Ollama] Connection failed")
                self.is_available = False
        except Exception as e:
            print(f"[Ollama] Not available: {e}")
            self.is_available = False

    def inference(self, query: str, controls: dict, context=None, enable_wiki=False):
        """
        Refine caption with LLM. Always returns a valid result.
        """
        # If too many errors or not available, return original
        if not self.is_available or self.error_count >= self.max_errors:
            return {
                "raw_caption": query,
                "caption": query,
                "wiki": "",
            }

        prompt = f"""{self.system_prompt}

Refine this image description:
"{query}"

Refined description:
"""

        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        refined_text = query

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower for more deterministic/accurate output
                    "num_predict": 80,  # More tokens for detailed descriptions
                    "top_p": 0.9,  # Nucleus sampling for quality
                    "repeat_penalty": 1.1  # Avoid repetition
                }
            }

            response = requests.post(
                self.base_url,
                json=payload,
                timeout=180,  # Increased timeout
            )

            self.last_request_time = time.time()

            if response.status_code == 200:
                result = response.json()
                refined_text = result.get("response", query).strip()

                # Clean up response
                if refined_text.startswith('"') and refined_text.endswith('"'):
                    refined_text = refined_text[1:-1]
                
                # Remove common prefixes
                prefixes = ["Here is the refined description:", "Refined description:", "Here's the refined description:"]
                for prefix in prefixes:
                    if refined_text.lower().startswith(prefix.lower()):
                        refined_text = refined_text[len(prefix):].strip()

                # Show full output
                word_count = len(refined_text.split())
                print(f"[Ollama] Refined ({word_count}w): {refined_text}")
                
                # Reset error count on success
                self.error_count = 0
                
            elif response.status_code == 404:
                self.is_available = False
                print(f"[Ollama] Model not found - disabled")
            else:
                self.error_count += 1
                print(f"[Ollama] Error {response.status_code} (attempt {self.error_count}/{self.max_errors})")

        except requests.exceptions.Timeout:
            self.error_count += 1
            print(f"[Ollama] Timeout (attempt {self.error_count}/{self.max_errors})")
        except requests.exceptions.ConnectionError:
            self.error_count += 1
            print(f"[Ollama] Connection error (attempt {self.error_count}/{self.max_errors})")
        except Exception as e:
            self.error_count += 1
            print(f"[Ollama] Error: {e} (attempt {self.error_count}/{self.max_errors})")

        return {
            "raw_caption": query,
            "caption": refined_text,
            "wiki": "",
        }

    def vision_inference(self, image_array, prompt="Describe this image in detail. Focus on the main object, its color, shape, and key visual features."):
        """
        Vision-based inference: Pass image directly to vision LLM (NO BLIP).
        
        Args:
            image_array: numpy array (RGB) of the cropped region
            prompt: text prompt to guide the vision model
            
        Returns:
            dict with 'caption' key containing the LLM's visual description
        """
        if not self.is_available or self.error_count >= self.max_errors:
            return {
                "caption": "an object",  # Fallback
            }

        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Convert numpy array to base64 image
            pil_image = Image.fromarray(image_array)
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Ollama vision API payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_base64],  # Image as base64
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 60,
                }
            }

            response = requests.post(
                self.base_url,
                json=payload,
                timeout=300,  # Increased for vision models
            )

            if response.status_code == 200:
                result = response.json()
                caption = result.get("response", "an object").strip()
                
                # Clean up response
                if caption.startswith('"') and caption.endswith('"'):
                    caption = caption[1:-1]
                
                return {"caption": caption}
            elif response.status_code == 500:
                # Server error - retry once after short delay
                print(f"[Ollama Vision] HTTP 500 - retrying...")
                import time
                time.sleep(2)
                
                # Retry
                response = requests.post(self.base_url, json=payload, timeout=300)
                if response.status_code == 200:
                    result = response.json()
                    caption = result.get("response", "an object").strip()
                    if caption.startswith('"') and caption.endswith('"'):
                        caption = caption[1:-1]
                    return {"caption": caption}
                else:
                    print(f"[Ollama Vision] Retry failed - HTTP {response.status_code}")
                    return {"caption": "an object"}
            else:
                print(f"[Ollama Vision] HTTP {response.status_code}")
                return {"caption": "an object"}
                
        except Exception as e:
            self.error_count += 1
            print(f"[Ollama Vision] Error: {e}")
            return {"caption": "an object"}


# Legacy Groq-based refiner
class GroqTextRefiner:
    """Groq API-based text refiner."""
    
    def __init__(self, device, api_keys: List[str] = None):
        print(f"Initializing GroqTextRefiner on {device}")

        if api_keys is None:
            raw = os.environ.get("GROQ_API_KEYS", "")
            api_keys = [k.strip() for k in raw.split(",") if k.strip()]
        elif isinstance(api_keys, str):
            api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]

        if not api_keys:
            raise ValueError("No Groq API keys provided.")

        self.api_keys = api_keys
        self.key_index = 0
        self.model = "llama-3.3-70b-versatile"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.min_delay = 0.1
        self.last_request_time = 0
        self.system_prompt = "Refine image descriptions by identifying visible objects."

        print(f"[Groq] Loaded {len(self.api_keys)} API keys")

    def _current_key(self):
        return self.api_keys[self.key_index]

    def _rotate_key(self):
        self.key_index += 1
        if self.key_index >= len(self.api_keys):
            raise RuntimeError("All Groq API keys exhausted.")

    def inference(self, query: str, controls: dict, context=None, enable_wiki=False):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        refined_text = query

        while True:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._current_key()}",
                }

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Refine: {query}"},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200,
                }

                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=15,
                )

                self.last_request_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    refined_text = result["choices"][0]["message"]["content"].strip()
                    break

                if response.status_code in (401, 403, 429):
                    self._rotate_key()
                    continue

                break

            except Exception:
                self._rotate_key()
                continue

        return {
            "raw_caption": query,
            "caption": refined_text,
            "wiki": "",
        }
