import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login
# Load a small open-source Transformer model (TinyLlama)



def print_query_matrix(model_name: str, device: str = "cpu", layer_index: int = 0):

    try:
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device) #load in bf16 for speed and memory usage

        # Access the query weights.  The exact path may vary slightly depending on the model architecture.
        # We'll try the most common paths and handle potential errors.

        query_weights = None
        try:
            # gpt2, gpt-neo, gpt-j, pythia style
            query_weights = model.transformer.h[layer_index].attn.q_proj.weight
        except AttributeError:
            pass

        try:
            # Mistral, Llama style
            query_weights = model.model.layers[layer_index].self_attn.q_proj.weight
        except AttributeError:
          pass

        try:
            # OPT style
            query_weights = model.model.decoder.layers[layer_index].self_attn.q_proj.weight
        except AttributeError:
            pass


        if query_weights is None:
            raise ValueError(f"Could not find query weights at specified layer ({layer_index}) for model {model_name}.  Check model architecture and layer index.")


        print(f"Query Matrix (Weights) for Layer {layer_index} of {model_name}:")
        print(query_weights)
        print(f"\nShape: {query_weights.shape}") #useful for debugging


        #Example of getting a single row or column (optional):
        #print("\nExample: First Row:")
        #print(query_weights[0,:]) #First row

        #print("\nExample: First Column:")
        #print(query_weights[:,0])   #First column

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nCommon issues and troubleshooting:")
        print("- Ensure the model name is correct and exists on the Hugging Face Hub.")
        print("- Check your internet connection.")
        print("- If using a GPU, make sure CUDA is properly installed and configured.")
        print("- The layer index might be out of bounds.  Check the model's architecture for the number of layers.")
        print("- The model might not have a 'q_proj' attribute in the expected location.  Inspect the model's source code if necessary.")
        print("- If you are running out of memory reduce the layer_index, use a smaller model, or use a GPU with more memory.  bf16 also uses less memory")



# Example usage (replace with your desired model and layer):
model_name = "mistralai/Mistral-7B-v0.1"  # Example: A larger, more capable model.
#model_name = "EleutherAI/pythia-70m" # Example:  A smaller, faster model for testing.
#model_name = 'gpt2' #Another smaller, faster model
layer_index = 2  # Inspect the third layer (index 2)
login("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")
print_query_matrix(model_name, device="cpu", layer_index=layer_index)