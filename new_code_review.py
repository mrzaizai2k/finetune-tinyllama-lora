import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from typing import List, Dict, Union, Optional
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 device: str = None,
                 load_in_8bit: bool = True,
                 use_cache: bool = True):
        """
        Initialize the LLM service with configurable parameters.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
            load_in_8bit: Whether to use 8-bit quantization to reduce memory usage
            use_cache: Whether to use KV cache for faster generation
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.load_in_8bit = load_in_8bit
        
        try:
            logger.info(f"Loading tokenizer for {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure padding token if needed
            if self.tokenizer.pad_token is None:
                logger.info("Setting pad_token to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with proper configuration
            logger.info(f"Loading model {model_name} on {self.device}")
            quantization_config = {}
            if self.load_in_8bit and self.device != "cpu":
                quantization_config = {"load_in_8bit": True}
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device if self.device != "cpu" else None,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                use_cache=use_cache,
                **quantization_config
            )
            
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {str(e)}")
            raise
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """Determine the appropriate device to use."""
        if device is not None:
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return "mps"  # For Apple Silicon
        else:
            return "cpu"
    
    def format_prompt(self, user_input: str) -> str:
        """
        Format the user input into a prompt suitable for the model.
        Can be customized based on the specific model requirements.
        """
        # This formatting works for Llama-2-chat models
        if "llama-2" in self.model_name.lower():
            return f"User: {user_input}\nAssistant:"
        # Add other model-specific formatting as needed
        else:
            return f"User: {user_input}\nAssistant:"
    
    def extract_response(self, full_output: str, prompt: str) -> str:
        """
        Extract the model's response from the full output.
        Handles various edge cases in the response parsing.
        """
        try:
            # Remove the original prompt
            if prompt in full_output:
                response = full_output[len(prompt):].strip()
            # Fall back to splitting on "Assistant:" if prompt removal fails
            elif "Assistant:" in full_output:
                parts = full_output.split("Assistant:", 1)
                if len(parts) > 1:
                    response = parts[1].strip()
                else:
                    response = full_output
            else:
                response = full_output
                
            return response
        except Exception as e:
            logger.error(f"Error extracting response: {str(e)}")
            return full_output
    
    def generate_response(self, 
                         user_input: str,
                         max_length: int = 2048,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """
        Generate a response for a single user input.
        
        Args:
            user_input: The user's input text
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (True) or greedy decoding (False)
        
        Returns:
            Generated response text
        """
        # Input validation
        if not user_input or not isinstance(user_input, str):
            logger.warning(f"Invalid user input: {user_input}")
            return "I couldn't process that input. Please try again with a valid text query."
        
        # Format the prompt
        prompt = self.format_prompt(user_input)
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode output
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract response
            response = self.extract_response(output, prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"An error occurred while generating the response: {str(e)}"
        
    def batch_generate(self, 
                      user_inputs: List[str],
                      max_length: int = 2048,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      do_sample: bool = True) -> List[str]:
        """
        Generate responses for multiple user inputs in an efficient batch.
        
        Args:
            user_inputs: List of user input texts
            max_length: Maximum length of generated responses
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            List of generated response texts
        """
        if not user_inputs:
            return []
            
        # Format all prompts
        prompts = [self.format_prompt(user_input) for user_input in user_inputs]
        
        try:
            # Tokenize all inputs in a batch
            batch_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            
            # Move batch to the correct device
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            
            # Generate outputs
            with torch.no_grad():
                output_ids = self.model.generate(
                    **batch_inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode all outputs
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Extract responses
            responses = [self.extract_response(output, prompt) 
                        for output, prompt in zip(outputs, prompts)]
            
            return responses
            
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            return [f"An error occurred in batch processing: {str(e)}"] * len(user_inputs)
    
    def clear_cuda_cache(self):
        """Clear CUDA cache to free up memory."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("CUDA cache cleared")
    
    def __del__(self):
        """Clean up resources when the service is destroyed."""
        try:
            self.clear_cuda_cache()
            logger.info("Resources cleaned up")
        except:
            pass


# Example usage
if __name__ == "__main__":
    # Create service with error handling
    try:
        service = LLMService(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            load_in_8bit=True
        )
        
        # Process a single query
        print("\nSingle query example:")
        response = service.generate_response(
            "What is machine learning?",
            max_length=512,  # Shorter for demo purposes
            temperature=0.7
        )
        print(response)
        
        # Process multiple queries efficiently in batch
        print("\nBatch processing example:")
        batch_queries = [
            "What is deep learning?",
            "Explain natural language processing.",
            "How do transformers work?"
        ]
        
        responses = service.batch_generate(
            batch_queries,
            max_length=512,  # Shorter for demo
            temperature=0.7
        )
        
        for i, resp in enumerate(responses):
            print(f"\nQuery {i+1}: {batch_queries[i]}")
            print(resp)
            print("-" * 50)
            
        # Clean up resources
        service.clear_cuda_cache()
        
    except Exception as e:
        print(f"Error running example: {str(e)}")