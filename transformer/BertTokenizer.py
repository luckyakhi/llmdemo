from transformers import BertTokenizer

def tokenize_with_pretrained_wordpiece(prompt, vocab_file="tokenizer/vocab.txt"):
    """
    Tokenizes an input prompt using a pre-trained WordPiece tokenizer.

    Args:
        prompt: The input text prompt (string).
        vocab_file: Path to the vocabulary file (string).
                     Defaults to "vocab.txt".

    Returns:
        A tuple containing:
            - tokens: A list of token strings.
            - token_ids: A list of token IDs (integers).
            - tokenizer: The tokenizer object.
        Returns None if there's an error.
    """
    try:
        # Initialize the tokenizer
        tokenizer = BertTokenizer(vocab_file=vocab_file)

        # Tokenize the prompt
        encoding = tokenizer.encode(prompt)  # Returns a BatchEncoding object

        tokens = encoding.tokens  # Access tokens from the BatchEncoding object
        token_ids = encoding.ids  # Access ids from the BatchEncoding object

        return tokens, token_ids, tokenizer

    except Exception as e:
        print(f"Error during tokenization: {e}")
        return None


# Example usage
prompt = "This is an example using a pre-trained WordPiece tokenizer. Coding is fun!"

result = tokenize_with_pretrained_wordpiece(prompt)

if result:
    tokens, token_ids, tokenizer = result
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    decoded_text = tokenizer.decode(token_ids)
    print("Decoded Text:", decoded_text)

    coding_encoding = tokenizer.encode("coding")
    print("\n'coding' tokens:", coding_encoding.tokens)

else:
    print("Tokenization failed.")