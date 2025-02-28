import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig


def generate_final_embeddings(prompt, model_name="bert-base-uncased", reduced_dim=10):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        original_dim = config.hidden_size  # 768 for BERT

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer embeddings

        # Reduce dimensions for embeddings
        projection_layer = torch.nn.Linear(original_dim, reduced_dim)
        semantic_embeddings = projection_layer(hidden_states).squeeze(0)
        positional_encodings = projection_layer(generate_positional_encodings(len(input_ids[0]), original_dim))
        final_embeddings = semantic_embeddings + positional_encodings

        # Value vectors remain in original 768-dim space
        value_vectors = outputs.hidden_states[-4].squeeze(0)

        # Get token names
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

        # Project last "bank" token
        top_5_tokens_bank = project_to_vocab(value_vectors, tokens, tokenizer, model)

        # Predict next token
        next_token_predictions = predict_next_token(input_ids, attention_mask, tokenizer, model)

        # Save both projections
        save_vocab_projection("embeddings2/vocab_projection.csv", top_5_tokens_bank, next_token_predictions)

        return input_ids, final_embeddings, tokenizer

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


def project_to_vocab(value_vectors, tokens, tokenizer, model):
    """Projects the value vector of the last 'bank' token onto the vocabulary."""
    target_idx = [i for i, token in enumerate(tokens) if token == "bank"]
    if not target_idx:
        return []

    last_bank_idx = target_idx[-1]
    value_vector = value_vectors[last_bank_idx]  # Extract last "bank" value vector (768-dim)

    # Get word embedding matrix
    word_embeddings = model.get_input_embeddings().weight  # Shape: (Vocab_size, 768)

    # Project to vocab space (logits)
    logits = torch.matmul(word_embeddings, value_vector)  # Shape: (Vocab_size,)

    # Compute softmax probabilities
    softmax_probs = F.softmax(logits, dim=0)

    # Get top 5 tokens
    top_5_indices = torch.topk(softmax_probs, 5).indices.tolist()
    top_5_tokens = [(tokenizer.convert_ids_to_tokens(idx), logits[idx].item(), softmax_probs[idx].item()) for idx in top_5_indices]

    return top_5_tokens


def predict_next_token(input_ids, attention_mask, tokenizer, model):
    """Predicts the next token based on the given prompt."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

    # Get logits for the last token position
    last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

    # Compute softmax probabilities
    softmax_probs = F.softmax(last_token_logits, dim=0)

    # Get top 5 next tokens
    top_5_indices = torch.topk(softmax_probs, 5).indices.tolist()
    top_5_next_tokens = [(tokenizer.convert_ids_to_tokens(idx), last_token_logits[idx].item(), softmax_probs[idx].item()) for idx in top_5_indices]

    return top_5_next_tokens


def save_vocab_projection(filename, top_5_tokens_bank, top_5_next_tokens):
    """Saves the projection of the last 'bank' token and predicted next token probabilities to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Token", "Logit", "Softmax Probability", "Type"])

        # Log "bank" projection
        for token, logit, softmax in top_5_tokens_bank:
            writer.writerow([token, f"{logit:.5f}", f"{softmax:.5f}", "Bank Token Projection"])

        # Log next token prediction
        for token, logit, softmax in top_5_next_tokens:
            writer.writerow([token, f"{logit:.5f}", f"{softmax:.5f}", "Next Token Prediction"])


# Example usage
prompt = "They deposited money in the bank and went to the river bank for"
model_name = "bert-base-uncased"
reduced_dim = 10

result = generate_final_embeddings(prompt, model_name, reduced_dim)
if result:
    input_ids, final_embeddings, tokenizer = result
    print("\nFinal Embeddings shape:", final_embeddings.shape)
    print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids.squeeze(0)))
    print("Top 5 projected tokens & next token predictions saved to vocab_projection.csv")
else:
    print("Failed to generate embeddings.")
