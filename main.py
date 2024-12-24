from recommender_functionality import (
    get_user_input,
    match_show_names,
    load_pickle_file,
    calculate_average_vector,
    initialize_usearch_index,
    load_embeddings_to_index,
    get_top_n_closest_shows_with_usearch,
    calculate_percentages,
    get_inspired_for_one_show,
    image_generation_from_prompt,
    EMBEDDINGS_FILE
)

def main():
    # Step 1: Get user input
    while True:
        user_shows = get_user_input()

        # Load available show titles
        embeddings = load_pickle_file(EMBEDDINGS_FILE)
        if not embeddings:
            print("Could not load show embeddings. Exiting.")
            return

        available_shows = list(embeddings.keys())

        # Step 2: Match user input with available shows
        matched_shows = match_show_names(user_shows, available_shows)
        print(f"Making sure, do you mean {', '.join(matched_shows)}?(y/n)")
        user_confirmation = input().strip().lower()
        if user_confirmation == 'y':
            break
        else:
            print("Sorry about that. Let's try again. Please make sure to write the names of the TV shows correctly.")

    print("Great! Generating recommendations now...")

    # Step 3: Generate recommendations
    average_vector = calculate_average_vector(matched_shows)
    if average_vector is None:
        print("Unable to calculate recommendations due to missing embeddings. Exiting.")
        return

    # Initialize and load the usearch index
    index = initialize_usearch_index(embeddings)
    show_titles = load_embeddings_to_index(EMBEDDINGS_FILE, index)

    # Get top 5 recommendations
    top_shows = get_top_n_closest_shows_with_usearch(average_vector, show_titles, index, matched_shows, top_n=5)
    percentages = calculate_percentages(top_shows)

    # Step 4: Display the top recommendations
    if percentages:
        print("Here are the top TV shows I think you would love:")
        for show, score in percentages:
            print(f"{show} ({int(score)}%)")
    else:
        print("Could not generate recommendations. Exiting.")
        return

   # Step 5: Generate an inspired TV show
    inspired_show_1 = get_inspired_for_one_show(matched_shows)
    inspired_show_2 = get_inspired_for_one_show(matched_shows)
    if inspired_show_1 and inspired_show_2:
        print("\nI have also created two unique show ideas based on your preferences:")
        print(inspired_show_1)
        print(inspired_show_2)

        # Extract the title from the inspired show
        def extract_title(show_idea):
            if " - " in show_idea:
                return show_idea.split(" - ")[0].strip()
            return None

        title_show_1 = extract_title(inspired_show_1)
        title_show_2 = extract_title(inspired_show_2)

        if title_show_1 and title_show_2:
            # Step 6: Generate and display an image for the inspired show
            prompt_show_1 = f"Ad poster for a TV show: {title_show_1}"
            prompt_show_2 = f"Ad poster for a TV show: {title_show_2}"
            # image_generation_from_prompt(prompt_show_1)
            # image_generation_from_prompt(prompt_show_2)
        else:
            print("Error: Failed to extract titles for the inspired shows.")
    else:
        print("Failed to generate an inspired TV show.")
    
if __name__ == "__main__":
    main()
