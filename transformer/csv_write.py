import csv

def write_embeddings_to_csv(embeddings_2d, reduced_dim, csv_filename="embeddings.csv"):
    """
    Writes embeddings to a CSV file with token names as the first column.

    Args:
        embeddings_2d: A 2D list or tensor of embeddings.
        reduced_dim: The dimensionality of the embeddings.
        csv_filename: The name of the CSV file to save the embeddings to.
    """
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"D{i}" for i in range(reduced_dim)])  # Column headers

            for row in embeddings_2d.tolist():  # Convert each row to a list
                writer.writerow(row)

    except Exception as e:
        print(f"Error writing embeddings to CSV: {e}")