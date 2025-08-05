import os
import tkinter as tk
from tkinter import filedialog
from subprocess import Popen, PIPE, CalledProcessError, run
from srt_from_json import create_srt_from_json
import logging

def run_in_conda_env(command, env_name="whisperx"):
    """Run a command inside a specified Conda environment."""
    try:
        activate_script = r"C:\Users\phili\anaconda3\Scripts\activate.bat"
        command_str = f"{activate_script} {env_name} && "
        command_str += " ".join([f'"{arg}"' if " " in arg else arg for arg in command])

        process = Popen(command_str, stdout=PIPE, stderr=PIPE, shell=True, text=True)
        output, error = process.communicate()

        if process.returncode != 0:
            raise CalledProcessError(process.returncode, command, output=output, stderr=error)

        print(output)

    except CalledProcessError as e:
        print(f"Error executing command in {env_name} environment: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise

def create_ass_from_whisperx_json(json_path, ass_path):
    """Converts WhisperX JSON output to ASS subtitle file."""
    try:
        conversion_script_path = "./ass_from_json.py"
        command = [
            "python", conversion_script_path,
            "--input", json_path,
            "--output", ass_path
        ]
        print(f"Running conversion script: {command}")
        result = run(command, check=True, capture_output=True, text=True)
        print(result.stdout)

    except CalledProcessError as e:
        print(f"Error occurred while running the conversion script:")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise

def main():
    # Initialize Tkinter and hide the root window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to select the source video file
    video_file = filedialog.askopenfilename(
        title="Select the Source Video File",
        filetypes=(("Video Files", "*.mp4;*.mkv;*.avi"), ("All Files", "*.*"))
    )

    if not video_file:
        print("No file selected. Exiting.")
        return

    # Generate derived paths, preserving the directory
    video_dir = os.path.dirname(video_file)
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    json_output = os.path.join(video_dir, f"{base_name}.json")
    ass_output = os.path.join(video_dir, f"{base_name}.ass")
    final_video_output = os.path.join(video_dir, f"{base_name}_subtitled.mp4").replace(os.sep, '/')

    print(f"Video File: {video_file}")
    print(f"JSON Output: {json_output}")
    print(f"ASS Output: {ass_output}")
    print(f"Final Video Output: {final_video_output}")

    # Configure logging
    log_file = os.path.join(video_dir, "process.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Prompt the user for the number of speakers
 #   num_speakers = input("Enter the number of speakers (1 or 2): ").strip()

    # Prompt the user for the language
 #   language = input("Enter the language (de or en): ").strip()

    # Run WhisperX to generate the JSON file
    whisperx_command = [
        "whisperx",
        video_file,
        "--model", "large-v3",
#        "--language", language,
        "--language=en",
        "--highlight_words=True",
#        "--max_line_count=1",
#        "--max_line_width=20",
        "--diarize",
        "--hf_token", "%HF_TOKEN%",
        "--output_dir", video_dir
    ]

 #   if num_speakers == "1":
 #       whisperx_command.insert(7, "--max_line_width=42")
 #      whisperx_command.insert(8, "42")

    logging.info("Running WhisperX...")
    try:
        run_in_conda_env(whisperx_command)
    except CalledProcessError as e:
        logging.error(f"WhisperX execution failed: {e}")
        return  # Exit if WhisperX fails

    # Generate the ASS file
    logging.info("Generating ASS file...")
    create_ass_from_whisperx_json(json_output, ass_output)

    # Generate the SRT file (new addition)
    logging.info("Generating SRT file...")
    create_srt_from_json(json_output, video_dir)  # Pass the directory, not the full path

    # Burn the subtitles into the video using FFmpeg
    logging.info("Burning subtitles into video...")
    # Change the current working directory
    original_cwd = os.getcwd()  # Store the original working directory
    os.chdir(video_dir)  # Change to the video's directory
    
    exit()

    # Construct the FFmpeg command using relative paths
    ffmpeg_command = [
        "ffmpeg",
        "-i", os.path.basename(video_file),  # Use relative path for input
        "-vf", f"subtitles='{os.path.basename(ass_output)}'",  # Use relative path for subtitles
        "-c:a", "copy",
        os.path.basename(final_video_output)  # Use relative path for output
    ]

    try:
        run(ffmpeg_command, check=True)
    finally:
        os.chdir(original_cwd)  # Restore the original working directory

    logging.info(f"Process complete! Subtitled video saved at {final_video_output}")

if __name__ == "__main__":
    main()