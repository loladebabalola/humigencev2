"""
LLM module for RAG pipeline.

Supports LLaMA (GGUF), Mistral, Phi-2 with GPU VRAM validation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import psutil

logger = logging.getLogger(__name__)

def _rag_debug() -> bool:
    return os.environ.get("RAG_DEBUG", "0") in ("1", "true", "TRUE", "yes", "YES")

@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model_name: str = "microsoft/Phi-2"
    model_type: str = "local"  # "local", "api", "huggingface", "gguf"
    device: str = "auto"
    context_length: int = 8192  # Context window size for the model
    max_tokens: int = 512  # Maximum tokens to generate
    max_new_tokens: int = 512  # Maximum new tokens to generate (balanced for quality responses)
    temperature: float = 0.7  # Higher for more diverse, less repetitive responses
    top_p: float = 0.9
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    trust_remote_code: bool = False
    model_path: Optional[str] = None  # For local GGUF models
    quantized: bool = False  # For quantized models

class BaseLLM(ABC):
    """Abstract base class for LLM models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        pass

class VRAMValidator:
    """Validates GPU VRAM before model loading."""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, Any]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {"available": False, "total_memory": 0, "free_memory": 0}
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory
            
            return {
                "available": True,
                "total_memory": total_memory,
                "allocated_memory": allocated_memory,
                "free_memory": free_memory,
                "total_gb": total_memory / (1024**3),
                "free_gb": free_memory / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {"available": False, "total_memory": 0, "free_memory": 0}

class LocalLLM(BaseLLM):
    """Local LLM implementation with VRAM validation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the local LLM model with VRAM validation."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Determine device with VRAM validation
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info(f"✅ Using GPU for LLM")
                else:
                    self.device = "cpu"
                    logger.info("CUDA not available, using CPU")
            else:
                self.device = self.config.device
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load model with appropriate settings
            if self.device == "cuda":
                # GPU settings
                torch_dtype = torch.float16 if not self.config.quantized else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=self.config.trust_remote_code,
                    torch_dtype=torch_dtype,
                    device_map="auto" if torch.cuda.device_count() > 1 else 0,
                    low_cpu_mem_usage=True
                )
            else:
                # CPU settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=self.config.trust_remote_code,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self.model = self.model.to("cpu")
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loaded local LLM: {self.config.model_name} on {self.device}")
            
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load local LLM: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using local model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            # Use strong defaults for RAG generation
            max_new_tokens = kwargs.get("max_new_tokens", 512)
            temperature = kwargs.get("temperature", 0.2)
            top_p = kwargs.get("top_p", 0.9)
            stop = kwargs.get("stop", [])
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.2,  # Increased repetition penalty
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Apply stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0].strip()
            
            # Log raw output for debugging
            if _rag_debug():
                print("\n[LLM RAW OUTPUT]\n" + (generated_text[:2000] if generated_text else "<empty>") + "\n")
            
            # Fail-safe: ensure non-empty response
            if not generated_text or generated_text.strip() == "":
                return "I cannot answer based on the provided information."
            
            # Post-process to remove meta-reasoning
            processed_text = self._postprocess_answer(generated_text)
            return processed_text or "I cannot answer based on the provided information."
            
        except Exception as e:
            logger.error(f"LocalLLM generation failed: {e}")
            return "I cannot answer based on the provided information."
    
    def _postprocess_answer(self, text: str) -> str:
        """Post-process answer to remove meta-reasoning and ensure clean output."""
        if not text:
            return "I cannot answer based on the provided documents."
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip meta-reasoning lines and empty lines
            if not line:
                continue
            if any(line.startswith(prefix) for prefix in [
                "The user", "We need to", "I will", "Let me", "Based on",
                "Looking at", "From the context", "The context says",
                "I can see", "It appears", "It seems", "According to",
                "The document", "The text", "The information"
            ]):
                continue
            filtered_lines.append(line)
        
        result = '\n'.join(filtered_lines).strip()
        
        # Ensure we have a meaningful answer
        if not result or len(result) < 10:
            return "I cannot answer based on the provided documents."
        
        # Remove any trailing meta-commentary
        if result.endswith(('.', '!', '?')):
            return result
        else:
            return result + '.'
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                results.append("")
        return results

class LLMManager:
    """Manages LLM models and operations with intelligent selection."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or self._get_default_config()
        self.llm = self._create_llm()
    
    def _get_default_config(self) -> LLMConfig:
        """Get default LLM configuration."""
        return LLMConfig(
            model_name="microsoft/Phi-2",
            model_type="local",
            device="auto",
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
    
    def _create_llm(self) -> BaseLLM:
        """Create appropriate LLM based on config with intelligent selection."""
        if self.config.model_type == "local":
            return LocalLLM(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    @staticmethod
    def _score_model_name(name: str) -> int:
        """Score model name to prefer instruct-tuned models."""
        n = name.lower()
        score = 0
        if "instruct" in n or "chat" in n or "assistant" in n:
            score += 3
        if "llama-3" in n or "llama3" in n or "qwen2" in n or "mistral" in n:
            score += 2
        if "gguf" in n:
            score += 1
        return score

    @staticmethod
    def detect_best_model() -> LLMConfig:
        """Automatically detect and configure the best available model."""
        # For now, return a default configuration
        return LLMConfig(
            model_name="microsoft/Phi-2",
            model_type="local",
            device="auto"
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        return self.llm.generate(prompt, **kwargs)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        return self.llm.generate_batch(prompts, **kwargs)
    
    def update_config(self, config: LLMConfig):
        """Update LLM configuration."""
        self.config = config
        self.llm = self._create_llm()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model."""
        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "device": getattr(self.config, 'device', 'unknown'),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }