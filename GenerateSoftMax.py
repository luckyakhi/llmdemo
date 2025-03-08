import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig


def generate_final_embeddings(prompt, model_name="bert-base-uncased", reduced_dim=10):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        original_dim = config.hidden_size

        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer embeddings
            attention = outputs.attentions[-1]  # Last layer attention

        # Reduce dimensions
        projection_layer = torch.nn.Linear(original_dim, reduced_dim)
        semantic_embeddings = projection_layer(hidden_states).squeeze(0)
        positional_encodings = projection_layer(generate_positional_encodings(len(inputs.input_ids[0]), original_dim))
        final_embeddings = semantic_embeddings + positional_encodings

        # Extract key, query, value vectors
        key_vectors = projection_layer(outputs.hidden_states[-2].squeeze(0))
        query_vectors = projection_layer(outputs.hidden_states[-3].squeeze(0))
        value_vectors = projection_layer(outputs.hidden_states[-4].squeeze(0))

        # Get token names
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze(0))

        # Compute softmax scores for the last occurrence of "bank"
        softmax_scores = compute_softmax_scores(semantic_embeddings, tokens, "bank")

        # Save embeddings
        save_to_csv("embeddings2/semantic_embeddings.csv", tokens, semantic_embeddings)
        save_to_csv("embeddings2/positional_encodings.csv", tokens, positional_encodings)
        save_to_csv("embeddings2/final_embeddings.csv", tokens, final_embeddings)
        save_to_csv("embeddings2/key_vectors.csv", tokens, key_vectors)
        save_to_csv("embeddings2/query_vectors.csv", tokens, query_vectors)
        save_to_csv("embeddings2/value_vectors.csv", tokens, value_vectors)
        save_to_csv("embeddings2/softmax_scores.csv", tokens, softmax_scores)

        return inputs.input_ids, final_embeddings, tokenizer

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def generate_positional_encodings(seq_len, dim):
    pos = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


def compute_softmax_scores(embeddings, tokens, target_token):
    target_indices = [i for i, token in enumerate(tokens) if token == target_token]
    if not target_indices:
        return torch.zeros_like(embeddings[:, 0])

    last_target_idx = target_indices[-1]  # Use last occurrence
    target_embedding = embeddings[last_target_idx]
    similarities = torch.matmul(embeddings, target_embedding)
    softmax_scores = F.softmax(similarities, dim=0)
    return softmax_scores.unsqueeze(1)  # To match CSV format


def save_to_csv(filename, tokens, embeddings):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Token"] + [f"Dim_{i + 1}" for i in range(embeddings.shape[1])])
        for token, emb in zip(tokens, embeddings.tolist()):
            writer.writerow([token] + [f"{val:.5f}" for val in emb])


# Example usage
prompt = "They deposited money in the bank and went to the river bank for"
model_name = "bert-base-uncased"
reduced_dim = 10

result = generate_final_embeddings(prompt, model_name, reduced_dim)
if result:
    input_ids, final_embeddings, tokenizer = result
    print("\nFinal Embeddings shape:", final_embeddings.shape)
    print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids.squeeze(0)))
else:
    print("Failed to generate embeddings.")
