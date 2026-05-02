import os
import sys
import time
import subprocess
import tempfile
from datetime import timedelta
import numpy as np
import soundfile as sf
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
    This version uses FFmpeg for conversion and soundfile for true streaming,
    ensuring low and constant memory usage.
    """

    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initializes the Transcriber.
        """
        print("Initializing transcription model...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Model '{model_size}' loaded on {device} with compute type {compute_type}.")
        # The model itself will consume memory, which is expected.
        # For 'base' on CPU, this is often 500-700MB.

    def _convert_to_wav(self, audio_path):
        """
        Converts any audio file to a temporary 16kHz mono WAV file using FFmpeg.
        Returns the path to the temporary file.
        """
        print("Converting audio to a temporary WAV file for processing...")
        # Create a temporary file with a .wav extension
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filepath = temp_file.name
        temp_file.close() # Close the file so ffmpeg can write to it

        # FFmpeg command to convert to 16kHz mono PCM WAV
        # -i: input file
        # -ar 16000: set audio sample rate to 16kHz
        # -ac 1: set audio channels to 1 (mono)
        # -c:a pcm_s16le: set audio codec to 16-bit PCM (standard for WAV)
        # -y: overwrite output file if it exists
        command = [
            "ffmpeg",
            "-i", audio_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-y",
            temp_filepath
        ]

        try:
            # Use DEVNULL to hide ffmpeg's verbose output from the console
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Conversion successful.")
            return temp_filepath
        except subprocess.CalledProcessError as e:
            print(f"Error during FFmpeg conversion: {e}", file=sys.stderr)
            # Clean up the empty temp file if conversion fails
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return None
        except FileNotFoundError:
            print("Error: ffmpeg is not installed or not in your system's PATH.", file=sys.stderr)
            print("Please install ffmpeg and ensure it's accessible from the command line.", file=sys.stderr)
            return None


    def transcribe_to_srt(self, audio_path, srt_path):
        """
        Transcribes an audio file and saves the output as an SRT file.
        """
        temp_wav_path = None
        try:
            # 1. Convert the original audio to a temporary WAV file
            temp_wav_path = self._convert_to_wav(audio_path)
            if not temp_wav_path:
                return # Conversion failed

            # 2. Open the WAV file with soundfile for chunked reading
            with sf.SoundFile(temp_wav_path, 'r') as audio_file:
                samplerate = audio_file.samplerate
                total_frames = audio_file.frames
                duration_s = total_frames / samplerate
                
                # Define chunk size in seconds (e.g., 30 seconds)
                chunk_duration_s = 30
                chunk_frames = chunk_duration_s * samplerate
                
                num_chunks = int(np.ceil(total_frames / chunk_frames))
                
                print(f"Audio loaded successfully. Total duration: {duration_s:.2f} seconds.")
                print(f"Processing in {num_chunks} chunks of up to {chunk_duration_s} seconds each.")
                
                srt_content = []
                segment_index = 1
                start_time = time.time()
                time_offset = 0.0

                # 3. Process the file chunk by chunk
                for i, chunk in enumerate(audio_file.iter(chunk_size=chunk_frames)):
                    print(f"Transcribing chunk {i + 1}/{num_chunks}...")
                    
                    # soundfile gives integer array, convert to float32
                    audio_np = chunk.astype(np.float32)
                    
                    # Normalize if it's not already
                    if np.max(np.abs(audio_np)) > 1.0:
                         audio_np /= np.iinfo(chunk.dtype).max
                    
                    # Transcribe the chunk
                    segments, _ = self.model.transcribe(audio_np, beam_size=5)

                    for segment in segments:
                        start_sec = time_offset + segment.start
                        end_sec = time_offset + segment.end
                        
                        srt_content.append(str(segment_index))
                        srt_content.append(f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}")
                        srt_content.append(segment.text.strip())
                        srt_content.append("")
                        
                        segment_index += 1
                    
                    time_offset += chunk_duration_s

            # 4. Write the SRT content to a file
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_content))
            
            print("-" * 50)
            print(f"Transcription complete! SRT file saved to: {srt_path}")
            end_time = time.time()
            print(f"Total processing time: {(end_time - start_time):.2f} seconds.")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            # 5. Clean up the temporary WAV file
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
                print(f"Temporary file {temp_wav_path} has been deleted.")


if __name__ == "__main__":
    transcriber = AudioFileTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8"
    )

    try:
        input_path = input("Please enter the full path to your audio file: ").strip().strip('"')

        if not os.path.isfile(input_path):
            print(f"Error: The file '{input_path}' was not found.")
            sys.exit(1)

        directory, filename = os.path.split(input_path)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base_name}.srt")
        
        transcriber.transcribe_to_srt(input_path, output_path)

    except KeyboardInterrupt:
        print("\nTranscription cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during setup: {e}")