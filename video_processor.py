import os
import tkinter as tk
from tkinter import filedialog
import json
import logging
from subprocess import run, CalledProcessError
import gc
import whisperx
from inputimeout import inputimeout, TimeoutOccurred
from whisperx.diarize import DiarizationPipeline

# Import your custom subtitle creation functions
from srt_from_json import create_srt_from_json
from ass_from_json import create_ass_from_json

# Define constants for supported file types to ensure consistency
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.aac', '.flac', '.m4a']
VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov']

def process_video_to_subtitles(video_file):
    """
    Full pipeline to transcribe a video/audio file and generate subtitles.
    """
    if not video_file:
        print("No file selected. Exiting.")
        return

    # --- 1. Setup Paths and Logging ---
    video_dir = os.path.dirname(video_file)
    base_name = os.path.splitext(os.path.basename(video_file))[0]

    # Path for the initial transcript before alignment and diarization
    initial_json_output = os.path.join(video_dir, f"{base_name}_initial.json")
    # Path for the final, processed transcript with speaker info
    final_json_output = os.path.join(video_dir, f"{base_name}_final.json")
    ass_output = os.path.join(video_dir, f"{base_name}.ass")
    final_video_output = os.path.join(video_dir, f"{base_name}_subtitled.mp4")
    log_file = os.path.join(video_dir, "process.log")

    print(f"Processing file: {video_file}")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info(f"Starting processing for {video_file}")

    # --- 2. Transcription with WhisperX ---
    device = "cuda"
    batch_size = 16
    compute_type = "float16"
    hf_token = os.environ.get("HF_TOKEN") # Use environment variable for Hugging Face token
    if not hf_token:
        message = "Hugging Face token not found. Please set the HF_TOKEN environment variable."
        logging.error(message)
        print(f"ERROR: {message}")
        return

    try:
        # Load audio
        logging.info("Loading audio...")
        audio = whisperx.load_audio(video_file)

        # Transcribe
        logging.info("Transcribing audio with whisper...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=batch_size)
        del model
        gc.collect()

        # Save initial transcript for potential manual editing
        with open(initial_json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"Initial transcript saved to {initial_json_output}")

        # --- 3. Optional Manual Intervention ---
        try:
            edit_prompt = inputimeout(
                prompt=f"You have 10 seconds to decide if you want to manually edit the transcript at:\n{initial_json_output}\nEdit now? (y/n): ",
                timeout=10
            )
        except TimeoutOccurred:
            edit_prompt = 'n'

        if edit_prompt.lower() == 'y':
            input("Please edit the JSON file, save it, and then press Enter to continue...")
            # Load the potentially edited file
            with open(initial_json_output, "r", encoding="utf-8") as f:
                result = json.load(f)
            logging.info("Reloaded edited transcript.")

        # --- 4. Align and Diarize ---
        # Align
        logging.info("Aligning transcript...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model_a
        gc.collect()

        # Diarize and assign speakers
        logging.info("Performing speaker diarization...")
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_model
        gc.collect()

        # Save final processed data
        with open(final_json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"Final processed data with speaker info saved to {final_json_output}")

        # --- 5. Generate Subtitle Files ---
        logging.info("Generating .ass subtitle file...")
        create_ass_from_json(final_json_output, ass_output)

        logging.info("Generating word-level .srt subtitle file...")
        create_srt_from_json(final_json_output, video_dir)

        # --- 6. Burn Subtitles with FFmpeg ---
        logging.info("Burning subtitles into video...")
        
        original_cwd = os.getcwd()
        try:
            # Change to the video's directory to avoid ffmpeg pathing issues on Windows
            os.chdir(video_dir)
            logging.info(f"Changed working directory to {video_dir} for ffmpeg processing.")

            # Use relative paths for ffmpeg, which is more robust
            relative_video_file = os.path.basename(video_file)
            relative_ass_output = os.path.basename(ass_output)
            relative_final_video_output = os.path.basename(final_video_output)

            # Check if the input is an audio file to construct the correct ffmpeg command
            file_extension = os.path.splitext(video_file)[1].lower()

            if file_extension in AUDIO_EXTENSIONS:
                # Input is audio: create a black video, add audio, and burn subtitles
                logging.info("Input is an audio file. Creating a black video with subtitles.")
                ffmpeg_command = [
                    "ffmpeg",
                    "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=25",
                    "-i", relative_video_file,
                    "-vf", f"subtitles='{relative_ass_output}'",
                    "-c:v", "libx264",
                    "-c:a", "copy",
                    "-shortest",
                    "-y",
                    relative_final_video_output
                ]
            else:
                # Input is a video: burn subtitles onto the existing video
                logging.info("Input is a video file. Burning subtitles into the existing video.")
                ffmpeg_command = [
                    "ffmpeg",
                    "-i", relative_video_file,
                    "-vf", f"subtitles='{relative_ass_output}'",
                    "-c:a", "copy",
                    "-y",
                    relative_final_video_output
                ]

            run(ffmpeg_command, check=True, capture_output=True, text=True)
            logging.info(f"Process complete! Subtitled video saved at {final_video_output}")
            print(f"\nSuccess! Subtitled video created at: {final_video_output}")

        finally:
            # Always change back to the original directory
            os.chdir(original_cwd)
            logging.info(f"Restored original working directory: {original_cwd}")

    except FileNotFoundError as e:
        logging.error(f"A file was not found: {e}")
        print(f"ERROR: A file was not found. Check the logs at {log_file}")
    except CalledProcessError as e:
        logging.error(f"FFmpeg failed with exit code {e.returncode}")
        logging.error(f"FFmpeg stdout: {e.stdout}")
        logging.error(f"FFmpeg stderr:\n{e.stderr}")
        print(f"ERROR: FFmpeg failed. Check the logs at {log_file}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred. Check the logs at {log_file}")


def main():
    """
    Main function to run the script.
    """
    root = tk.Tk()
    root.withdraw()

    # Create file type strings for the dialog from the constants
    video_types_str = ";".join([f"*{ext}" for ext in VIDEO_EXTENSIONS])
    audio_types_str = ";".join([f"*{ext}" for ext in AUDIO_EXTENSIONS])
    media_types_str = f"{video_types_str};{audio_types_str}"

    video_file = filedialog.askopenfilename(
        title="Select the Source Video or Audio File",
        filetypes=(
            ("Media Files", media_types_str),
            ("Video Files", video_types_str),
            ("Audio Files", audio_types_str),
            ("All Files", "*.*")
        )
    )
    process_video_to_subtitles(video_file)


if __name__ == "__main__":
    main()