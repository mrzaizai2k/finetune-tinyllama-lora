import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMService:
    def __init__(self):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model.to("cuda")
    
    def generate_response(self, user_input):
        # Format the prompt for a chat model
        prompt = f"User: {user_input}\nAssistant:"
        
        # Tokenize the input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Return everything after "Assistant:"
        return output.split("Assistant:")[1].strip()
    
    def batch_generate(self, user_inputs):
        responses = []
        for user_input in user_inputs:
            responses.append(self.generate_response(user_input))
        return responses

# Example usage
if __name__ == "__main__":
    service = LLMService()
    
    # Process a single query
    response = service.generate_response("What is machine learning?")
    print(response)
    
    # Process multiple queries
    responses = service.batch_generate([
        "What is deep learning?",
        "Explain natural language processing.",
        "How do transformers work?"
    ])
    
    for resp in responses:
        print(resp)
        print("-" * 50)