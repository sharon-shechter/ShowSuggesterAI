# Video Search Engine

## Overview
This project implements a video search engine that allows users to search for specific content inside a video using captions generated from scenes. The project leverages AI tools and libraries to break down a video into scenes, generate textual descriptions, and enable efficient search functionality.



## Video - 
[Watch the demo video](https://drive.google.com/file/d/1rpg4m3sKiCXhRhL3IL9Le6X8LxyKYpx2/view?usp=sharing)



## Features
- **Download Video:** Uses `yt-dlp` to download a video from YouTube based on a search query.
- **Scene Detection:** Uses `pyscenedetect` to split the video into different scenes.
- **Scene Captioning:** Uses `Moondream2` to generate textual descriptions for each detected scene.
- **Search Functionality:** Allows users to search for specific words in the scene captions using `RapidFuzz` for fuzzy matching.
- **Auto-Complete Search:** Uses `prompt_toolkit` to provide search suggestions based on available captions.
- **Collage Generation:** Displays and saves a collage of matching scene images.
- **Video-Based Search:** Optionally uses Google Gemini's video understanding model for broader queries.

## Workflow
1. **Downloading a Video**
   - The program will search YouTube for "Super Mario Movie Trailer" and download it using `yt-dlp`.
   - If the video is already downloaded, this step will be skipped.

2. **Scene Detection**
   - The video is split into 50-80 scenes using `pyscenedetect`.
   - Scene images are stored in a designated folder.

3. **Generating Captions**
   - Captions for each scene are generated using `moondream2`.
   - A JSON file (`scene_captions.json`) maps scene numbers to their captions.
   - If the JSON file exists, this step is skipped.

4. **Search in Video**
   - The user is prompted: `Search the video using a word:`
   - The search runs first using simple string matching, then using `RapidFuzz` for better accuracy.
   - Matching scenes are displayed as a collage.

5. **Auto-Complete Feature**
   - While typing a query, `prompt_toolkit` suggests words from the captions.

6. **Video Model Search**
   - The user can choose to search the video using a multimodal model (Google Gemini).
   - The model processes the entire video and finds relevant frames.
   - Matching frames are extracted and displayed as a collage.

## Output
- The program saves:
  - Extracted scene images in a folder.
  - `scene_captions.json` with captions mapped to scenes.
  - `collage.png` showing scenes matching the search query.

## Notes
- Ensure your OpenAI/Gemini API key is set up if using the video search option.
- Use smaller videos for testing to reduce processing time.

## Contribution
Feel free to submit issues and pull requests to improve the project!

## License
MIT License

