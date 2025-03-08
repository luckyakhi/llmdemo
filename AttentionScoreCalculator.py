import torch
from transformers import AutoTokenizer, AutoModel

def calculate_attention_scores(prompt, target_token):
    """Calculates attention scores, handling subtokens correctly."""

    model_name = "bert-base-uncased"  # Or any other suitable model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    encoded_input = tokenizer(prompt, return_tensors="pt")

    target_token_ids = tokenizer.convert_tokens_to_ids(target_token)

    if isinstance(target_token_ids, int):  # Single token
        target_token_ids = [target_token_ids]  # Make it a list

    target_token_indices = []

    for target_token_id in target_token_ids:
        token_indices = (encoded_input["input_ids"][0] == target_token_id).nonzero(as_tuple=False)
        if len(token_indices)>0:
            target_token_indices.append(token_indices[0].item())
        else:
            print(f"Target subtoken '{tokenizer.convert_ids_to_tokens(target_token_id)}' not found in the vocabulary.")
            return None

    if not target_token_indices:
        print(f"Target token '{target_token}' not found in the vocabulary.")
        return None



    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state.squeeze(0)

    query = hidden_states[target_token_indices].mean(dim = 0).unsqueeze(0) # Averaging query vectors for subtokens
    key = hidden_states
    value = hidden_states

    attention_scores = torch.matmul(query, key.transpose(0, 1)) / (key.shape[-1] ** 0.5)
    attention_weights = torch.softmax(attention_scores, dim=-1).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])

    token_attention_dict = {}
    for i, token in enumerate(tokens):
        token_attention_dict[token] = attention_weights[i].item()

    return token_attention_dict



# Example usage (demonstrating subtokens):
prompt = "This is a test with subtokens."
target_token = "subtokens"  # Will be split into "sub" and "##tokens"
attention_scores = calculate_attention_scores(prompt, target_token)

if attention_scores:
    for token, score in attention_scores.items():
        print(f"Attention score for '{token}': {score}")


prompt = "This is a test with a missing token."
target_token = "missing"
attention_scores = calculate_attention_scores(prompt, target_token)

if attention_scores:
    for token, score in attention_scores.items():
        print(f"Attention score for '{token}': {score}")

prompt = "The cat sat on the mat."
target_token = "cat"
attention_scores = calculate_attention_scores(prompt, target_token)

if attention_scores:
    for token, score in attention_scores.items():
        print(f"Attention score for '{token}': {score}")