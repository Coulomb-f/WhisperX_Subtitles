import json
import os

def create_srt_from_json(json_path, output_dir):
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            segments = data["segments"]
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        exit(1)

    try:
        srt_lines = []

        for segment in segments:
            words = segment["words"]

            for word_info in words:
                if "start" not in word_info or "end" not in word_info:
                    print(f"Skipping word with missing timing information: {word_info['word']}")
                    continue

                word_start = word_info["start"]
                word_end = word_info["end"]
                speaker = word_info.get("speaker", "UNKNOWN")  # Get speaker, default to "UNKNOWN"

                start_srt = f"{int(word_start // 3600):02d}:{int((word_start % 3600) // 60):02d}:{int(word_start % 60):02d},{int((word_start % 1) * 1000):03d}"
                end_srt = f"{int(word_end // 3600):02d}:{int((word_end % 3600) // 60):02d}:{int(word_end % 60):02d},{int((word_end % 1) * 1000):03d}"

                srt_lines.append(f"{start_srt} --> {end_srt}\n[{speaker}]: {word_info['word']}\n\n")

        base_name = os.path.splitext(os.path.basename(json_path))[0]
        srt_path = os.path.join(output_dir, f"{base_name}_word_lvl.srt")
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            for i, subtitle in enumerate(srt_lines, start=1):
                srt_file.write(f"{i}\n")
                srt_file.write(subtitle)
        print(f"SRT file created: {srt_path}")
    except Exception as e:
        print(f"An error occurred: {e}")