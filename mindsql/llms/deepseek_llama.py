# mindsql/llms/deepseek_llama.py

from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from mindsql.llms.base import BaseLLM

def load_deepseek_model(model_path: str):
    """
    Load the DeepSeek-R1-Distill-Llama-8B model and tokenizer.
    :param model_path: Path to the model (local or Hugging Face model ID).
    :return: Loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

class DeepSeekLlama(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DeepSeek-R1-Distill-Llama-8B model.
        :param config: Configuration dictionary containing necessary parameters like model path.
        """
        self.config = config
        self.model, self.tokenizer = load_deepseek_model(config["model_path"])

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the DeepSeek-R1-Distill-Llama-8B model.
        :param prompt: The input prompt for the model.
        :param kwargs: Additional parameters for generation (e.g., temperature, max_tokens).
        :return: The generated response as a string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, **kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
