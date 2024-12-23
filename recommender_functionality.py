from thefuzz import process
from apiKeys import OPEN_AI_API_KEY
import pickle
import os
import numpy as np
from scipy.spatial import distance




# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the embeddings file
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embedded _shows.pkl")




def get_user_input():
    """
    Prompt the user to input TV shows they loved watching.
    """
    user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show: ")
    return [show.strip() for show in user_input.split(",")]

def match_show_names(user_shows, available_shows):
    """
    Match user-inputted TV show names to available show names using fuzzy matching.

    :param user_shows: List of user-provided TV show names
    :param available_shows: List of valid TV show names
    :return: A list of matched show names
    """
    matched_shows = []
    for show in user_shows:
        match, confidence = process.extractOne(show, available_shows)
        matched_shows.append(match)  # Only save the matched name
    return matched_shows


def load_pickle_file(file_path):
    """
    Load a pickle file and return the data.
    :param file_path: Path to the pickle file
    :return: Deserialized object
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def calculate_average_vector(matched_shows):
    """
    Calculate the average vector for the matched shows.

    :param matched_shows: List of matched show titles
    :return: Average vector as a numpy array
    """
    vectors = []
    embeddings = load_pickle_file(EMBEDDINGS_FILE)
    for show in matched_shows:
        if show in embeddings:
            vectors.append(np.array(embeddings[show]))
        else:
            print(f"Embedding not found for show: {show}")
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        print("No valid embeddings found to calculate average.")
        return None


def distances_from_avg_vector(average_vector, distance_metric="cosine"):
    """
    Return the distances between an average embedding and all embeddings.

    :param average_vector: The embedding to compare (average vector)
    :param distance_metric: The distance metric to use (default is cosine)
    :return: List of distances
    """
    distance_metrics = {
        "cosine": distance.cosine,
        "L1": distance.cityblock,
        "L2": distance.euclidean,
        "Linf": distance.chebyshev,
    }

    # Load all embeddings from the pickle file
    embeddings = list(load_pickle_file(EMBEDDINGS_FILE).values())

    # Select the distance metric function
    metric_function = distance_metrics.get(distance_metric, distance.cosine)

    # Compute distances
    distances = [metric_function(average_vector, embedding) for embedding in embeddings]
    return distances


def get_top_n_closest_shows(distances, show_titles, top_n=5):
    """
    Get the top N TV shows closest to the average vector.

    :param distances: List of distances between the average vector and all TV show embeddings
    :param show_titles: List of TV show titles corresponding to the embeddings
    :param top_n: Number of closest shows to return (default is 5)
    :return: List of tuples (TV show title, distance) for the top N closest shows
    """
    if len(distances) != len(show_titles):
        raise ValueError("The number of distances must match the number of show titles.")

    # Get indices of the smallest distances
    closest_indices = np.argsort(distances)[:top_n]

    # Get the corresponding show titles and distances
    closest_shows = [(show_titles[i], distances[i]) for i in closest_indices]

    return closest_shows

