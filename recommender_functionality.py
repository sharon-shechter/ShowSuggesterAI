from thefuzz import process
import pickle
import os
import numpy as np
import requests
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
import time 
from tkinter import Tk, Label
from usearch.index import Index
import openai
from PIL import Image, ImageTk
from io import BytesIO
from apiKeys import OPEN_AI_API_KEY , LIGHTX_API_KEY



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


# Initialize the usearch index
index = Index(
    ndim=300,  # Set this to match the dimensionality of your embeddings
    metric="cos",  # Use cosine similarity
    dtype="f32"    # Floating-point precision
)

def load_embeddings_to_index(embeddings_file):
    """
    Load embeddings from a pickle file and populate the usearch index.

    :param embeddings_file: Path to the pickle file containing embeddings
    """
    embeddings = load_pickle_file(embeddings_file)
    show_titles = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()), dtype=np.float32)

    # Add embeddings to the index
    keys = np.arange(len(show_titles))  # Unique integer keys for each show
    index.add(keys, vectors)
    return show_titles

# Initialize the usearch index dynamically
def initialize_usearch_index(embeddings):
    """
    Initializes the usearch index based on the dimensions of the embeddings.
    :param embeddings: Dictionary of TV show embeddings
    :return: Initialized usearch Index
    """
    sample_vector = next(iter(embeddings.values()))
    dim = len(sample_vector)
    index = Index(
        ndim=dim,  # Use the dimension of the embeddings
        metric="cos",  # Cosine similarity
        dtype="f32"    # Floating-point precision
    )
    return index

def load_embeddings_to_index(embeddings_file, index):
    """
    Load embeddings from a pickle file and populate the usearch index.

    :param embeddings_file: Path to the pickle file containing embeddings
    :param index: Usearch index object
    """
    embeddings = load_pickle_file(embeddings_file)
    show_titles = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()), dtype=np.float32)

    # Batch add embeddings to the index
    keys = np.arange(len(show_titles))  # Unique integer keys for each show
    index.add(keys, vectors)
    return show_titles

def get_top_n_closest_shows_with_usearch(average_vector, show_titles, index, user_shows, top_n=5):
    """
    Get the top N TV shows closest to the average vector using the usearch index.

    :param average_vector: The average vector computed from matched shows
    :param show_titles: List of TV show titles corresponding to the embeddings
    :param index: Usearch index object
    :param user_shows: List of user's input shows to exclude from recommendations
    :param top_n: Number of closest shows to return (default is 5)
    :return: List of tuples (TV show title, distance) for the top N closest shows
    """
    # Perform the search
    matches = index.search(average_vector, top_n + len(user_shows))  # Fetch extra matches

    # Retrieve the show titles and distances, excluding user's input shows
    closest_shows = [
        (show_titles[match.key], match.distance)
        for match in matches
        if show_titles[match.key] not in user_shows
    ]

    # Return the top N closest shows
    return closest_shows[:top_n]

def calculate_percentages(top_shows, threshold=10):
    """
    Convert distances into percentages for each show and filter recommendations.

    :param top_shows: List of tuples (TV show title, distance)
    :param threshold: Minimum percentage threshold to include a recommendation (default is 10%)
    :return: Filtered list of tuples (TV show title, percentage score)
    """
    if not top_shows:
        return []

    distances = [dist for _, dist in top_shows]
    min_distance = min(distances)
    max_distance = max(distances)

    # Ensure meaningful scaling of percentages
    percentages = [
        (show, 100 * (1 - (dist - min_distance) / (max_distance - min_distance)))
        if max_distance != min_distance else (show, 100)  # Handle edge case where all distances are equal
        for show, dist in top_shows
    ]

    # Apply threshold to filter out very low scores
    filtered_percentages = [(show, round(score)) for show, score in percentages if score >= threshold]

    # Sort by percentage in descending order
    filtered_percentages.sort(key=lambda x: x[1], reverse=True)

    return filtered_percentages


def display_recommendations(percentages):
    """
    Format and display recommendations.

    :param percentages: List of tuples (TV show title, percentage score)
    """
    print("Top Recommendations:")
    for show, score in percentages:
        print(f"{show} ({int(score)}%)")


def get_inspired_for_one_show(tv_shows):
    """
    Generate a unique TV show idea based on a list of existing TV shows.
    """
    openai.api_key = OPEN_AI_API_KEY

    # Prepare the prompt for the OpenAI API
    prompt = (
        "Here is a list of TV shows: \n"
        f"{', '.join(tv_shows)}\n"
        "Based on these shows, create a unique and creative TV show idea. "
        "Provide the title and a brief description show."
        "DO like this format:"
        "Roommate Roulette (NO BOLD) -When a quirky group of mismatched roommates in a co-living space..."
    )

    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Using the 4.0 mini model
            messages=[
                {"role": "system", "content": "You are a creative TV show writer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7
        )

        # Extract the response text
        result = response['choices'][0]['message']['content'].strip()
        return result

    except Exception as e:
        print(f"Error generating TV shows: {e}")
        return None
    


#######################
# image generation code
#######################



# Replace with your LightX API Key
BASE_URL = "https://api.lightxeditor.com/external/api/v1"

def generate_image(prompt):
    """
    Generate an image using the LightX AI API.
    :param prompt: The text prompt for the image.
    :return: The orderId for the image generation request, or None if it fails.
    """
    url = f"{BASE_URL}/text2image"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": LIGHTX_API_KEY
    }
    data = {
        "textPrompt": prompt
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            if response_json and "body" in response_json and "orderId" in response_json["body"]:
                print(f"Image generation request successful for prompt: {prompt}")
                return response_json["body"]["orderId"]
            else:
                print(f"Unexpected response format or missing 'orderId': {response_json}")
                return None
        else:
            print(f"Failed to generate image. Status code: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error while making the API request: {e}")
        return None

def check_status(order_id):
    """
    Check the status of the image generation.
    :param order_id: The order ID of the image generation request.
    :return: The output URL if the image is ready, otherwise None.
    """
    url = f"{BASE_URL}/order-status"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": LIGHTX_API_KEY
    }
    payload = {
        "orderId": order_id
    }
    retries = 5
    for attempt in range(retries):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            body = response.json()["body"]
            if body["status"] == "active":
                return body["output"]
            elif body["status"] == "failed":
                print(f"Image generation failed for order ID: {order_id}")
                return None
        else:
            print(f"Failed to check status. Status code: {response.status_code}")
            print(response.text)
        time.sleep(3)  # Wait 3 seconds before retrying
    print("Image generation timed out.")
    return None

def show_image_in_tkinter(image_url):
    """
    Display an image inside a Tkinter window.
    :param image_url: The URL of the image to display.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code == 200:
            img_data = BytesIO(response.content)
            img = Image.open(img_data)

            # Create a Tkinter window
            window = tk.Tk()
            window.title("Generated Image")

            # Resize the image to fit in the window
            img = img.resize((500, 500), Image.LANCZOS)  # Updated resizing method
            img_tk = ImageTk.PhotoImage(img)

            # Add the image to the Tkinter window
            label = tk.Label(window, image=img_tk)
            label.image = img_tk  # Keep a reference to avoid garbage collection
            label.pack()

            # Run the Tkinter event loop
            window.mainloop()
        else:
            print(f"Failed to fetch image. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error displaying the image: {e}")


def image_generation_from_prompt(prompt):
    """
    Generate an image and display it inside the Python program.
    :param prompt: The prompt for generating the image.
    """
    # Generate image
    order_id = generate_image(prompt)
    
    if order_id:
        print(f"Order ID: {order_id}")
        
        # Check status and get image URL
        image_url = check_status(order_id)
        
        if image_url:
            print(f"Image URL: {image_url}")
            # Show the image inside the Python program
            show_image_in_tkinter(image_url)
        else:
            print("Failed to retrieve the image URL.")
    else:
        print("Image generation failed.")
