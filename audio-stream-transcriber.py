import os
import sys
import time
from datetime import timedelta
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel

# --- Helper Function for SRT Formatting ---
def format_srt_time(seconds):
    """Converts seconds to SRT time format (HH:MM:SS,ms)."""
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# --- Main Transcription Class ---
class AudioFileTranscriber:
    """
    A class to transcribe large audio files to SRT format using faster-whisper.
    
    Processes audio in chunks to manage memory usage effectively.
    """

    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initializes the Transcriber.

        Args:
            model_size (str): The size of the Whisper model (e.g., "base", "small", "medium").
            device (str): The device to run the model on ("cpu" or "cuda").
            compute_type (str): The computation type for the model ("int8", "float16").
        """
        print("Initializing transcription model...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Model '{model_size}' loaded on {device} with compute type {compute_type}.")

    def transcribe_to_srt(self, audio_path, srt_path):
        """
        Transcribes an audio file and saves the output as an SRT file.

        Args:
            audio_path (str): Path to the input audio file.
            srt_path (str): Path to save the output SRT file.
        """
        try:
            print(f"Loading audio file: {audio_path}")
            # pydub can handle various audio formats (mp3, wav, m4a, etc.)
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return

        # Ensure audio is 16kHz mono, which is required by Whisper
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        
        # Define chunk size in milliseconds (e.g., 30 seconds)
        chunk_length_ms = 30 * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        srt_content = []
        segment_index = 1
        total_chunks = len(chunks)
        start_time = time.time()
        
        print(f"Audio loaded successfully. Total duration: {len(audio) / 1000:.2f} seconds.")
        print(f"Processing in {total_chunks} chunks of up to {chunk_length_ms / 1000} seconds each.")

        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i + 1}/{total_chunks}...")
            
            # Convert pydub chunk to numpy array for faster-whisper
            # The audio data is normalized to be between -1 and 1
            audio_np = np.array(chunk.get_array_of_samples(), dtype=np.float32) / 32768.0

            # Transcribe the chunk
            segments, _ = self.model.transcribe(audio_np, beam_size=5)

            # Calculate the time offset for this chunk
            chunk_start_time_s = (i * chunk_length_ms) / 1000.0

            for segment in segments:
                start_sec = chunk_start_time_s + segment.start
                end_sec = chunk_start_time_s + segment.end
                
                # Format for SRT file
                srt_content.append(str(segment_index))
                srt_content.append(f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}")
                srt_content.append(segment.text.strip())
                srt_content.append("")  # Blank line separator
                
                segment_index += 1

        # Write the SRT content to a file
        try:
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_content))
            print("-" * 50)
            print(f"Transcription complete! SRT file saved to: {srt_path}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Total processing time: {processing_time:.2f} seconds.")

        except IOError as e:
            print(f"Error writing to file {srt_path}: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # You can customize the model size and hardware acceleration here
    # Model sizes: "tiny", "base", "small", "medium", "large-v2"
    # For GPU acceleration, change device="cuda" and compute_type="float16"
    transcriber = AudioFileTranscriber(
        model_size="base", 
        device="cpu", 
        compute_type="int8"
    )

    # Prompt user for the audio file path
    try:
        input_path = r"D:\Audio\Audiobooks\John Steinbeck - East Of Eden\John Steinbeck - East Of Eden  pt.1.mp3"

        # Simple validation: check if the file exists
        if not os.path.isfile(input_path):
            print(f"Error: The file '{input_path}' was not found.")
            sys.exit(1)

        # Determine the output SRT file path
        directory, filename = os.path.split(input_path)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base_name}.srt")
        
        # Start the transcription process
        transcriber.transcribe_to_srt(input_path, output_path)

    except KeyboardInterrupt:
        print("\nTranscription cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")