import openai
import pandas as pd
import pickle

# Path to your CSV file
CSV_FILE = r"C:\Users\sharon shechter\Desktop\school\Third year\Dudu\2\ShowSuggesterAI\imdb_tvshows - imdb_tvshows.csv"
PICKLE_FILE = r"C:\Users\sharon shechter\Desktop\school\Third year\Dudu\2\ShowSuggesterAI\embeddings.pkl"

def load_tv_shows(csv_file):
    """
    Load TV shows from a CSV file.

    :param csv_file: Path to the CSV file
    :return: A DataFrame of TV shows
    """
    return pd.read_csv(csv_file)

def save_embeddings(embeddings, pickle_file):
    """
    Save embeddings to a pickle file.

    :param embeddings: Dictionary of embeddings
    :param pickle_file: Path to the pickle file
    """
    with open(pickle_file, "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings(pickle_file):
    """
    Load embeddings from a pickle file if it exists.

    :param pickle_file: Path to the pickle file
    :return: Dictionary of embeddings
    """
    try:
        with open(pickle_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def generate_embeddings(api_key, shows_df, pickle_file):
    """
    Generate embeddings for each TV show and save them.

    :param api_key: OpenAI API key
    :param shows_df: DataFrame containing TV shows and their descriptions
    :param pickle_file: Path to the pickle file for storing embeddings
    """
    openai.api_key = api_key
    embeddings = load_embeddings(pickle_file)

    for _, row in shows_df.iterrows():
        show_name = row["Title"]  # Adjusted for capitalized column name
        show_description = row["Description"]  # Adjusted for capitalized column name

        if show_name not in embeddings:
            print(f"Generating embedding for {show_name}...")
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=show_description
            )
            embeddings[show_name] = response["data"][0]["embedding"]

    save_embeddings(embeddings, pickle_file)
    print(f"All embeddings saved to {pickle_file}.")
    

if __name__ == "__main__":
    # Load shows from CSV
    tv_shows = load_tv_shows(CSV_FILE)

    # Ensure the required columns are present
    if "Title" not in tv_shows.columns or "Description" not in tv_shows.columns:
        raise ValueError("CSV file must contain 'Title' and 'Description' columns.")

    # Your OpenAI API key


    # Generate embeddings
    generate_embeddings(OPEN_AI_API_KEY, tv_shows, PICKLE_FILE)
