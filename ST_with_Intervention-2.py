import os
import tkinter as tk
import json
from tkinter import filedialog
import logging
import whisperx
import gc
from subprocess import Popen, PIPE, CalledProcessError, run
from srt_from_json import create_srt_from_json
from whisperx.utils import WriteSRT, WriteTXT  # Import the WriteSRT and WriteTXT classes directly
from inputimeout import inputimeout, TimeoutOccurred
from whisperx.diarize import DiarizationPipeline
from whisperx.transcribe import get_writer  # Import the get_writer function

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
    # Define the correct paths
    base_dir = "C:/Users/phili/Documents/VS_CodePlayground/ass_subtitles"
    json_output = os.path.join(base_dir, "output.json")
    ass_output = os.path.join(base_dir, "output.ass")
    video_dir = os.path.join(base_dir, "videos")
    video_file = os.path.join(video_dir, "video.mp4")

    # Open file dialog to select the source video file
    video_file = filedialog.askopenfilename(
        title="Select the Source Video File",
        filetypes=(("Video Files", "*.mp4;*.mkv;*.avi;*.mov;*.aac;*.mp3;*.wav"), ("All Files", "*.*"))
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

    # Set device and other parameters
    device = "cuda"
    batch_size = 16
    compute_type = "float16"
    hf_token = os.environ.get("HF_TOKEN") # Use environment variable for Hugging Face token
    if not hf_token:
        message = "Hugging Face token not found. Please set the HF_TOKEN environment variable."
        logging.error(message)
        print(f"ERROR: {message}")
        return

    # 1. Transcribe with original Whisper
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(video_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"])  # before alignment

    # Save the initial transcription result to JSON for manual editing
    with open(json_output, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    # Ask the user if they want to edit the JSON file
    try:
        edit_json = inputimeout(prompt="Do you want to edit the JSON file? (y/n): ", timeout=3)
    except TimeoutOccurred:
        edit_json = 'n'

    if edit_json.lower() == 'y':
        input("Please edit the JSON file if needed and press Enter to continue...")

    # Load the edited JSON file
    with open(json_output, "r", encoding="utf-8") as json_file:
        result = json.load(json_file)

    # Ensure the 'language' key exists in the result dictionary
    if 'language' not in result:
        result['language'] = 'de'  # Set the language manually if not present

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    print(result["segments"])  # after alignment

    # 3. Assign speaker labels
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"])  # segments are now assigned speaker IDs

    # Save the final result to JSON
    with open(json_output, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    # Generate the ASS file
    logging.info("Generating ASS file...")
    create_ass_from_whisperx_json(json_output, ass_output)

    # Define the options dictionary (extracted from transcribe.py CLI parameters)
    options = {
        "highlight_words": True,  # Example option, add more as needed
        "max_line_width": 42,  # Default value from transcribe.py
        "max_line_count": 1,  # Default value from transcribe.py
        "highlight_color": "yellow",  # Default value from transcribe.py
        # Add other options here
    }

    # Ensure the 'language' key exists in the result dictionary before generating the SRT file
    if 'language' not in result:
        result['language'] = 'en'  # Set the language manually if not present

    # Use get_writer to get the writer instance
    writer = get_writer("srt", video_dir)
    writer_args = {arg: options[arg] for arg in ["highlight_words", "max_line_count", "max_line_width"]}

    # Generate the Word Level SRT file
    logging.info("Generating Word Level SRT file...")
    create_srt_from_json(json_output, video_dir)  # Pass the directory, not the full path

    # Generate the SRT file using the writer instance
    logging.info("Generating SRT file...")
    with open(os.path.join(video_dir, f"{base_name}.srt"), "w", encoding="utf-8") as file:
        writer(result, file, writer_args)  # Call the writer function directly

    # Generate the TXT file using WriteTXT class directly
    logging.info("Generating TXT file...")
    txt_writer = WriteTXT(output_dir=video_dir)  # Provide the output_dir argument
    with open(os.path.join(video_dir, f"{base_name}.txt"), "w", encoding="utf-8") as file:
        txt_writer.write_result(result, file, options)  # Call the write_result method

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