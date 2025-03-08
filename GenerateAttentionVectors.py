import csv
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


def generate_final_embeddings(prompt, model_name="bert-base-uncased", reduced_dim=50):
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        original_dim = config.hidden_size  # Typically 768 for models like GPT-2, BERT

        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        seq_len = inputs.input_ids.shape[1]

        # Get semantic embeddings from model's input embeddings layer
        with torch.no_grad():
            input_embeddings_layer = model.get_input_embeddings()
            semantic_embeddings = input_embeddings_layer(inputs.input_ids)  # (1, seq_len, original_dim)

        # Generate positional encodings
        positional_encodings = generate_positional_encodings(seq_len, original_dim).unsqueeze(
            0)  # (1, seq_len, original_dim)

        # Reduce dimensionality of semantic and positional encodings
        semantic_projection = torch.nn.Linear(original_dim, reduced_dim)
        positional_projection = torch.nn.Linear(original_dim, reduced_dim)
        reduced_semantic_embeddings = semantic_projection(semantic_embeddings)  # (1, seq_len, reduced_dim)
        reduced_positional_encodings = positional_projection(positional_encodings)  # (1, seq_len, reduced_dim)

        # Compute final embeddings
        final_embeddings = reduced_semantic_embeddings + reduced_positional_encodings  # (1, seq_len, reduced_dim)

        # Extract key, query, and value vectors from attention layers
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            last_hidden_state = outputs.last_hidden_state  # (1, seq_len, original_dim)

            key_projection = torch.nn.Linear(original_dim, reduced_dim)
            query_projection = torch.nn.Linear(original_dim, reduced_dim)
            value_projection = torch.nn.Linear(original_dim, reduced_dim)

            key_vectors = key_projection(last_hidden_state)  # (1, seq_len, reduced_dim)
            query_vectors = query_projection(last_hidden_state)  # (1, seq_len, reduced_dim)
            value_vectors = value_projection(last_hidden_state)  # (1, seq_len, reduced_dim)

        # Convert tensors to lists for CSV writing
        red_sem_list = reduced_semantic_embeddings.squeeze(0).tolist()
        red_pos_list = reduced_positional_encodings.squeeze(0).tolist()
        final_list = final_embeddings.squeeze(0).tolist()
        key_list = key_vectors.squeeze(0).tolist()
        query_list = query_vectors.squeeze(0).tolist()
        value_list = value_vectors.squeeze(0).tolist()

        # Convert token IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0].tolist())

        # Define CSV file names
        file_names = {
            "semantic": "embeddings/semantic_embeddings.csv",
            "positional": "embeddings/positional_encodings.csv",
            "final": "embeddings/final_embeddings.csv",
            "key": "embeddings/key_vectors.csv",
            "query": "embeddings/query_vectors.csv",
            "value": "embeddings/value_vectors.csv",
        }

        # Helper function to format values to 5 decimal places
        def format_values(values):
            return [format(v, ".5f") for v in values]

        # Write embeddings to CSV files
        for name, data_list in zip(file_names.keys(),
                                   [red_sem_list, red_pos_list, final_list, key_list, query_list, value_list]):
            with open(file_names[name], 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                header = ["Token"] + [f"{name.capitalize()}_Dim_{i + 1}" for i in range(reduced_dim)]
                writer.writerow(header)
                for token, values in zip(tokens, data_list):
                    writer.writerow([token] + format_values(values))

        return inputs.input_ids, final_embeddings, tokenizer

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def generate_positional_encodings(seq_len, dim):
    pos = torch.arange(seq_len).unsqueeze(1).float()  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


# Example usage
if __name__ == "__main__":
    prompt = "Deposit money in the bank and picnic near bank of the"
    model_name = "gpt2"  # Example: BERT with hidden size 768
    reduced_dim = 50  # Desired reduced dimension

    result = generate_final_embeddings(prompt, model_name, reduced_dim)

    if result:
        input_ids, final_embeddings, tokenizer = result
        print("\nFinal Embeddings shape:", final_embeddings.shape)
        print("Final Embeddings:", final_embeddings)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        print("Tokens:", tokens)
    else:
        print("Failed to generate embeddings.")
