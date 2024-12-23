import pytest
import numpy as np
from unittest.mock import patch
from scipy.spatial import distance
import os
from recommender_functionality import load_pickle_file, match_show_names, calculate_average_vector, distances_from_avg_vector,get_top_n_closest_shows

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the embeddings file
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embedded _shows.pkl")




def test_match_show_names():
    available_shows = [
        "Game Of Thrones",
        "Lupin",
        "The Witcher",
        "Breaking Bad",
        "Sherlock",
        "Dark"
    ]
    user_input = ["gem of throns", "lupan", "witcher"]
    valid_shows = match_show_names(user_input, available_shows=available_shows)

    # Expected result: tuples of matched show names with confidence scores
    expected_shows = [
        ('Game Of Thrones', 86),
        ('Lupin', 80),
        ('The Witcher', 90)
    ]
@patch("recommender_functionality.load_pickle_file")
def test_calculate_average_vector(mock_load_pickle_file):
    # Mock data: Embedding dictionary
    mock_embeddings = {
        "Game Of Thrones": [0.1, 0.2, 0.3],
        "Lupin": [0.4, 0.5, 0.6],
        "The Witcher": [0.7, 0.8, 0.9]
    }

    # Mock return value for load_pickle_file
    mock_load_pickle_file.return_value = mock_embeddings

    # Test input: Shows user liked
    matched_shows = ["Game Of Thrones", "Lupin", "The Witcher"]

    # Expected output
    expected_average = np.mean(
        [mock_embeddings["Game Of Thrones"], mock_embeddings["Lupin"], mock_embeddings["The Witcher"]],
        axis=0
    )

    # Call the function
    calculated_average = calculate_average_vector(matched_shows)

    # Assert the calculated average matches the expected average
    assert np.allclose(calculated_average, expected_average), (
        f"Expected {expected_average}, but got {calculated_average}"
    )


@patch("recommender_functionality.load_pickle_file")
def test_distances_from_avg_vector(mock_load_pickle_file):
    # Mock embeddings dictionary
    mock_embeddings = {
        "Game Of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Sherlock": [0.7, 0.8, 0.9]
    }
    mock_load_pickle_file.return_value = mock_embeddings

    # Define test inputs
    average_vector = [0.4, 0.5, 0.6]
    expected_distances = [
        distance.cosine(average_vector, mock_embeddings["Game Of Thrones"]),
        distance.cosine(average_vector, mock_embeddings["Breaking Bad"]),
        distance.cosine(average_vector, mock_embeddings["Sherlock"])
    ]

    # Call the function
    calculated_distances = distances_from_avg_vector(average_vector, distance_metric="cosine")

    # Assert distances match expected values
    assert np.allclose(calculated_distances, expected_distances), (
        f"Expected distances: {expected_distances}, but got: {calculated_distances}"
    )


def test_top5_closest_shows():
    distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    show_titles =["Game Of Thrones", "Breaking Bad", "Sherlock" , "Dark", "Lupin", "The Witcher"]
    

    # Expected top 5 closest shows
    expected_top5 = [
        ("Game Of Thrones", 0.1),
        ("Breaking Bad", 0.2),
        ("Sherlock", 0.3),
        ("Dark", 0.4),
        ("Lupin", 0.5),
    ]

    # Call the function
    top5 = get_top_n_closest_shows(distances, show_titles, 5)

    # Assert the top 5 closest shows match the expected result
    assert top5 == expected_top5

def test_fetch_data_from_dictionary():
    data = load_pickle_file(EMBEDDINGS_FILE)
    assert (list(data.keys())[0]).lower()   == "game of thrones"

def test_generate_recommendations():
    valid_shows = ["Game Of Thrones", "Lupin", "The Witcher"]
    recommendations = generate_recommendations(valid_shows)
    assert "Breaking Bad" in recommendations
    assert isinstance(recommendations["Breaking Bad"], float)
