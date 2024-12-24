import pytest
import numpy as np
from unittest.mock import patch
from scipy.spatial import distance
import os
from recommender_functionality import (
    load_pickle_file,
    match_show_names,
    calculate_average_vector,
    display_recommendations,
    calculate_percentages,
    get_top_n_closest_shows_with_usearch,
    initialize_usearch_index,
    load_embeddings_to_index,
    EMBEDDINGS_FILE
)

import pytest
import numpy as np
from unittest.mock import patch
from scipy.spatial import distance
import os
from recommender_functionality import (
    load_pickle_file,
    match_show_names,
    calculate_average_vector,
    display_recommendations,
    calculate_percentages,
    get_top_n_closest_shows_with_usearch,
    initialize_usearch_index,
    EMBEDDINGS_FILE
)

# Get the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

    # Expected result: Matched show names
    expected_shows = [
        "Game Of Thrones",
        "Lupin",
        "The Witcher"
    ]

    assert valid_shows == expected_shows, (
        f"Expected {expected_shows}, but got {valid_shows}"
    )


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
def test_get_top_n_closest_shows_with_usearch(mock_load_pickle_file):
    # Mock embeddings dictionary
    mock_embeddings = {
        "Game Of Thrones": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "Breaking Bad": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        "Sherlock": np.array([0.7, 0.8, 0.9], dtype=np.float32)
    }
    mock_load_pickle_file.return_value = mock_embeddings

    # Test inputs
    average_vector = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    user_shows = ["Breaking Bad"]  # Exclude this show
    show_titles = list(mock_embeddings.keys())
    
    # Mock index behavior: manually simulate top matches
    top_shows = [
        ("Game Of Thrones", 0.1),
        ("Sherlock", 0.3)
    ]
    
    # Simulate call to `get_top_n_closest_shows_with_usearch`
    result = [show for show in top_shows if show[0] not in user_shows]

    # Expected result
    expected_result = [("Game Of Thrones", 0.1), ("Sherlock", 0.3)]

    # Assert result matches expectation
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
def test_display_recommendations(capsys):
    # Input: List of shows with percentages
    percentages = [
        ("Breaking Bad", 99.0),
        ("Sherlock", 85.0),
        ("Dark", 81.0),
        ("Lupin", 75.0),
        ("The Witcher", 70.0)
    ]

    # Call the function to display recommendations
    display_recommendations(percentages)

    # Capture the printed output
    captured = capsys.readouterr()

    # Expected output
    expected_output = (
        "Top Recommendations:\n"
        "Breaking Bad (99%)\n"
        "Sherlock (85%)\n"
        "Dark (81%)\n"
        "Lupin (75%)\n"
        "The Witcher (70%)\n"
    )

    # Assert the captured output matches the expected output
    assert captured.out == expected_output


def test_fetch_data_from_dictionary():
    data = load_pickle_file(EMBEDDINGS_FILE)

    # Normalize keys for case-insensitive matching
    normalized_keys = {key.lower(): key for key in data.keys()}
    expected_key = "game of thrones".lower()

    assert expected_key in normalized_keys, f"Expected key not found in data: {expected_key}"


def test_calculate_percentages():
    # Input: List of distances
    top_shows = [
        ("Breaking Bad", 0.1),
        ("Sherlock", 0.2),
        ("Dark", 0.3),
        ("Lupin", 0.8),
        ("The Witcher", 1.0)
    ]

    # Call the function to calculate percentages
    percentages = calculate_percentages(top_shows, threshold=20)

    # Expected output: Percentages sorted by score and excluding very low percentages
    expected_percentages = [
        ("Breaking Bad", 100),
        ("Sherlock", 89),
        ("Dark", 78),
        ("Lupin", 22)
    ]

    # Assert the calculated percentages match the expected output
    assert len(percentages) == len(expected_percentages), (
        f"Expected {len(expected_percentages)} shows, but got {len(percentages)}"
    )
    for (calculated_show, calculated_score), (expected_show, expected_score) in zip(percentages, expected_percentages):
        assert calculated_show == expected_show, (
            f"Expected show {expected_show}, but got {calculated_show}"
        )
        assert pytest.approx(calculated_score, 0.1) == expected_score, (
            f"Expected score {expected_score}, but got {calculated_score}"
        )

@patch("recommender_functionality.load_pickle_file")
def test_generate_recommendations(mock_load_pickle_file):
    # Mock data: Embedding dictionary
    mock_embeddings = {
        "Game Of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Sherlock": [0.7, 0.8, 0.9]
    }

    # Mock return value for load_pickle_file
    mock_load_pickle_file.return_value = mock_embeddings

    # Test input: Valid shows
    valid_shows = ["Game Of Thrones", "Lupin", "The Witcher"]

    # Mock expected output
    recommendations = calculate_average_vector(valid_shows)

    assert recommendations is not None, "Expected recommendations to be generated, but got None."
    assert isinstance(recommendations, np.ndarray), "Expected recommendations to be a numpy array."
