"""
    Creates an ASS subtitle file with word-level and speaker-level highlighting
    from a WhisperX JSON output file. Handles files with or without speaker
    information.

    Args:
        json_path (str): Path to the WhisperX JSON file.
        ass_path (str): Path to save the generated ASS file.
"""

import json
import os
import argparse

def create_ass_from_json(json_path, ass_path):
    speaker_colors = {
        "SPEAKER_00": "&H128F07&",  # Green
        "SPEAKER_01": "&H702618&",  # Red
        "SPEAKER_02": "&H161691&",  # Blue
        "Extra": "&C9C967&"       # Yellow for extra speakers
    }

    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)  # Load the entire JSON object
            segments = data["segments"]   # Access the list of segments
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise

    try:
        with open(ass_path, "w", encoding="utf-8") as ass_file:
            # Write ASS header
            ass_file.write(
                """[Script Info]
Title: Word-Level Dynamic Highlighting
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,18,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
            )

            # Process each segment
            for segment in segments:
                full_text = segment["text"].strip()
                words = segment["words"]

                # Handle missing speaker information
                if "speaker" in segment:
                    speaker = segment["speaker"]

                    # Get speaker color
                    if speaker not in speaker_colors:
                        speaker = "Extra"  # Label any additional speaker as 'Extra'
                    color = speaker_colors[speaker]
                else:
                    color = "&H34495E&"  # Default color if no speaker information

                # Normalize spaces in full_text
                full_text = full_text.replace("\u00A0", " ")
                
                # Generate individual ASS dialogue lines for each word
                prev_word_end = None  # Keep track of the previous word's end time
                word_start_index = 0  # Keep track of the starting index for searching

                for i, word_info in enumerate(words):
                    # Skip words with missing timing information
                    if "start" not in word_info or "end" not in word_info:
                        print(f"Skipping word with missing timing information: {word_info['word']}")
                        continue

                    current_word = word_info["word"]
                    word_start = word_info["start"]
                    word_end = word_info["end"]

                    # Align start time with the end time of the previous word
                    if prev_word_end is not None:
                        word_start = prev_word_end  # Directly use the previous word's end time

                    # Normalize spaces in current_word
                    current_word = current_word.replace("\u00A0", " ")

                    # Find the index of the current word in the full text
                    try:
                        current_word_index = full_text.index(current_word, word_start_index)
                    except ValueError:
                        print(f"Warning: Word '{current_word}' not found in the segment.")
                        continue  # Skip this word

                    # Create the ASS dialogue line with the current word highlighted
                    ass_text = f"{full_text[:current_word_index]}{{\\1c{color}}}{{\\u1}}{current_word}{{\\u0}}{{\\1c&HFFFFFF&}}{full_text[current_word_index + len(current_word):]}"
                    
                    # Create ASS timestamps
                    start_ass = f"{int(word_start // 3600)}:{int((word_start % 3600) // 60):02}:{int(word_start % 60):02}.{int((word_start % 1) * 100):02}"
                    end_ass = f"{int(word_end // 3600)}:{int((word_end % 3600) // 60):02}:{int(word_end % 60):02}.{int((word_end % 1) * 100):02}"

                    # Write the dialogue line
                    ass_file.write(
                        f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{ass_text}\n"
                    )

                    # Update prev_word_end to the current word's end time
                    prev_word_end = word_end

                    # Update the starting index for the next word search
                    word_start_index = current_word_index + len(current_word)

        print(f"ASS file created: {ass_path}")

    except Exception as e:
        print(f"Error creating ASS file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output ASS file")
    args = parser.parse_args()

    # Access file paths from command-line arguments
    input_json_path = args.input
    output_ass_path = args.output

    create_ass_from_json(input_json_path, output_ass_path)