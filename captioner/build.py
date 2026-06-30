from .blip import BLIPCaptioner
from .base_captioner import BaseCaptioner
from .text_refiner import TextRefiner, GroqTextRefiner

def build_captioner(captioner_type, device, enable_filter=False):
    if captioner_type == 'blip':
        return BLIPCaptioner(device, enable_filter=enable_filter)
    else:
        raise NotImplementedError(f"Unknown captioner type: {captioner_type}")


def build_text_refiner(refiner_type, device, model="llama3", ollama_url=None, api_keys=None):
    """
    Build text refiner.
    
    Args:
        refiner_type: 'ollama' (local) or 'groq' (API)
        device: CUDA device
        model: Ollama model name (default: llama3)
        ollama_url: Ollama server URL (default: http://localhost:11434)
        api_keys: Groq API keys (only for 'groq' type)
    """
    if refiner_type in ('base', 'ollama'):
        # Default: Local Ollama LLM
        return TextRefiner(device, model=model, ollama_url=ollama_url)
    elif refiner_type == 'groq':
        # Legacy: Groq API
        if api_keys is not None:
            return GroqTextRefiner(device, api_keys)
        else:
            return GroqTextRefiner(device)
    else:
        raise NotImplementedError(f"Unknown text refiner type: {refiner_type}")
