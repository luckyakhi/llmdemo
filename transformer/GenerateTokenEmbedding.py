import csv
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from CommonUtils import format_values


def generate_final_embeddings(prompt, model_name="bert-base-uncased", reduced_dim=10):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        original_dim = config.hidden_size  # e.g., 768 for GPT-2

        inputs = tokenizer(prompt, return_tensors="pt")
        seq_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            input_embeddings_layer = model.get_input_embeddings()
            semantic_embeddings = input_embeddings_layer(inputs.input_ids)  # shape: (1, seq_len, original_dim)

        positional_encodings = generate_positional_encodings(seq_len, original_dim)  # shape: (seq_len, original_dim)
        positional_encodings = positional_encodings.unsqueeze(0)  # shape: (1, seq_len, original_dim)

        # Reduce dimensions for both semantic and positional embeddings if needed
        if original_dim != reduced_dim:
            semantic_projection = torch.nn.Linear(original_dim, reduced_dim)
            positional_projection = torch.nn.Linear(original_dim, reduced_dim)
            reduced_semantic_embeddings = semantic_projection(semantic_embeddings)  # shape: (1, seq_len, reduced_dim)
            reduced_positional_encodings = positional_projection(
                positional_encodings)  # shape: (1, seq_len, reduced_dim)
        else:
            reduced_semantic_embeddings = semantic_embeddings
            reduced_positional_encodings = positional_encodings

        # Compute final embeddings as the sum of the reduced semantic and positional parts
        final_embeddings = reduced_semantic_embeddings + reduced_positional_encodings  # shape: (1, seq_len, reduced_dim)

        # Remove the batch dimension to prepare for CSV writing
        red_sem_list = reduced_semantic_embeddings.squeeze(0).tolist()  # shape: (seq_len, reduced_dim)
        red_pos_list = reduced_positional_encodings.squeeze(0).tolist()  # shape: (seq_len, reduced_dim)
        final_list = final_embeddings.squeeze(0).tolist()  # shape: (seq_len, reduced_dim)

        # Convert token IDs to token strings for the first (and only) batch element
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0].tolist())

        # Define CSV file names
        semantic_csv_filename = "embeddings/semantic_embeddings.csv"
        positional_csv_filename = "embeddings/positional_encodings.csv"
        final_csv_filename = "embeddings/final_embeddings.csv"

        # Write semantic embeddings to CSV
        with open(semantic_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Token"] + [f"Dim_{i + 1}" for i in range(reduced_dim)]
            writer.writerow(header)
            for token, sem in zip(tokens, red_sem_list):
                writer.writerow([token] + format_values(sem))

        # Write positional encodings to CSV
        with open(positional_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Token"] + [f"Dim_{i + 1}" for i in range(reduced_dim)]
            writer.writerow(header)
            for token, pos in zip(tokens, red_pos_list):
                writer.writerow([token] + format_values(pos))

        # Write final embeddings to CSV
        with open(final_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Token"] + [f"Dim_{i + 1}" for i in range(reduced_dim)]
            writer.writerow(header)
            for token, fin in zip(tokens, final_list):
                writer.writerow([token] + format_values(fin))

        return inputs.input_ids, final_embeddings, tokenizer

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def generate_positional_encodings(seq_len, dim):
    pos = torch.arange(seq_len).unsqueeze(1).float()  # shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


# Example usage
if __name__ == "__main__":
    prompt = "Deposit money in the bank and picnic near bank of the"
    model_name = "gpt2"  # Example: GPT-2 with hidden size 768
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
