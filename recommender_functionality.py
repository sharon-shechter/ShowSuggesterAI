from thefuzz import process
from apiKeys import OPEN_AI_API_KEY
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
    :return: A list of tuples (matched_show, confidence)
    """
    matched_shows = []
    for show in user_shows:
        match, confidence = process.extractOne(show, available_shows)
        matched_shows.append((match, confidence))
    return matched_shows

if __name__ == "__main__":
    # Example list of available shows
    available_shows = [
        "Game Of Thrones",
        "Lupin",
        "The Witcher",
        "Breaking Bad",
        "Sherlock",
        "Dark"
    ]

    while True:
        user_shows = get_user_input()
        if len(user_shows) <= 1:
            print("Please enter more than one TV show.")
            continue

        matched_shows = match_show_names(user_shows, available_shows)

        # Confirm matches with the user
        confirmation = input(f"Making sure, do you mean {[match for match, _ in matched_shows]}? (y/n): ").strip().lower()

        if confirmation == 'y':
            print("Great! Proceeding with these shows:", [match for match, _ in matched_shows])
            break
        else:
            print("Sorry about that. Let's try again.")
