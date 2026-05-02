# 
# Audio Stream Transcriber
# Version 0.5
# 

import os
import sys
import time
import subprocess
import numpy as np
from datetime import timedelta
from faster_whisper import WhisperModel

# --- Helper Functions ---

def format_srt_time(seconds):
    """Converts seconds to SRT time format (HH:MM:SS,ms)."""
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def format_duration(seconds):
    """Formats a duration in seconds into a human-readable H:M:S string."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}h{minutes:02}m{secs:02}s"
    elif minutes > 0:
        return f"{minutes}m{secs:02}s"
    else:
        return f"{secs}s"

# --- Main Transcription Class ---

class AudioStreamTranscriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        print("Initializing transcription model...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Model '{model_size}' loaded on {device} with compute type {compute_type}.")

    def _get_audio_duration(self, audio_path):
        """Gets the total duration of the audio file using ffprobe."""
        command = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return float(result.stdout)
        except Exception as e:
            print(f"Error getting audio duration with ffprobe: {e}", file=sys.stderr)
            print("Please ensure ffprobe (part of FFmpeg) is installed and in your PATH.")
            return None

    def transcribe_to_srt(self, audio_path, srt_path):
        total_duration = self._get_audio_duration(audio_path)
        if total_duration is None: return

        print(f"Audio file duration: {format_duration(total_duration)}.")

        command = [
            "ffmpeg", "-i", audio_path, "-nostdin", "-threads", "0",
            "-f", "s16le", "-ac", "1", "-ar", "16000", "-",
        ]

        try:
            ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            
            samplerate = 16000
            block_duration_s = 60 
            block_size_bytes = block_duration_s * samplerate * 2 # I cannot remember why there's a `2` here.
            empty_fraction = 0.5

            segment_index = 1
            transcription_start_time = time.time()
            total_audio_processed_s = 0
            
            unprocessed_audio_buffer = np.array([], dtype=np.float32)
            
            this_is_the_last_block = False

            # Open the SRT file for writing at the beginning
            with open(srt_path, "w", encoding="utf-8") as srt_file:
                while True:
                    raw_audio_block = ffmpeg_process.stdout.read(block_size_bytes)
                    
                    if not raw_audio_block:
                        # So, if I let execution continue without my flag, I'd get to this clause,
                        if unprocessed_audio_buffer.size == 0:
                            break
                        #     and I'd miss this exit point because `unprocessed_audio_buffer.size != 0`
                        else:
                            # but I feel like I just don't know enough to confirm that I've done
                            #    everything because `unprocessed_audio_buffer` *might* be uncommitted
                            #    simply because it *might* belong with subsequent samples.
                            this_is_the_last_block = True
                            
                    last_committed_segment = None
                    
                    new_audio_block = np.frombuffer(raw_audio_block, dtype=np.int16).astype(np.float32) / 32768.0
                    current_block = np.concatenate([unprocessed_audio_buffer, new_audio_block])

                    segments, _ = self.model.transcribe(current_block, beam_size=5)
                    segments = list(segments)
                    
                    current_block_commit_point_s = 0
                    last_committed_segment_end_s = total_audio_processed_s
                    
                    # Determine which segments to commit
                    num_segments_to_commit = len(segments)
                    if raw_audio_block: # If we are not at the end of the file, don't commit the last segment
                        num_segments_to_commit -= 1
                    # Here's a spot where an `elif not num_segments_to_commit:` could break.
                    
                    if num_segments_to_commit > 0:
                        for i in range(num_segments_to_commit):
                            segment = segments[i]
                            start_sec = total_audio_processed_s + segment.start
                            end_sec = total_audio_processed_s + segment.end
                            
                            # Write segment directly to the SRT file
                            srt_file.write(f"{segment_index}\n")
                            srt_file.write(f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}\n")
                            srt_file.write(f"{segment.text.strip()}\n\n")
                            segment_index += 1
                        
                        srt_file.flush() # Ensure data is written to disk
                        
                        last_committed_segment = segments[num_segments_to_commit - 1]
                        current_block_commit_point_s = last_committed_segment.end
                        last_committed_segment_end_s = total_audio_processed_s + current_block_commit_point_s
                    
                    else: # There are no segments to commit
                        if raw_audio_block: # if this is not the last chunk of audio
                            # On this iteration of the loop, we read stuff, so its safe to loop.
                            last_committed_segment_end_s = total_audio_processed_s + block_duration_s*empty_fraction

                    # --- Real-time Progress Update ---
                    runtime = time.time() - transcription_start_time
                    processing_speed = last_committed_segment_end_s / runtime
                    remaining_audio_s = total_duration - last_committed_segment_end_s
                    
                    eta_s = 0
                    if processing_speed > 0:
                        eta_s = remaining_audio_s / processing_speed
                    
                    progress_line = (
                        f"runtime {format_duration(runtime)} | "
                        f"committed {format_duration(last_committed_segment_end_s)} / {format_duration(total_duration)} | "
                        f"ETA in {format_duration(eta_s)}"
                    )
                    # Print on one line, clearing the previous one
                    sys.stdout.write(f"\r\033[K{progress_line}")
                    sys.stdout.flush()
                    
                    if last_committed_segment != None:
                        print(f"\n~ {last_committed_segment.text.strip()}")
                    else:
                        print(f"\n~ ")
                    
                    # Update progress and buffer for the next iteration
                    total_audio_processed_s += current_block_commit_point_s
                    current_block_commit_point_samples = int(current_block_commit_point_s * samplerate)
                    unprocessed_audio_buffer = current_block[current_block_commit_point_samples:]
                    
                    # However, since I didn't have this check, and the flag set earlier, I'd still have uncommitted nothing.
                    
                    if this_is_the_last_block:
                        
                        # This would appear to indicate that my previous problem was (at least in part) due to
                        #     fractional remainders in the final audio block.
                        print(f"\n\nthis_is_the_last_block: {this_is_the_last_block}")
                        print(f"runtime: {runtime}")
                        print(f"remaining_audio_s: {remaining_audio_s}")
                        print(f"total_audio_processed_s: {total_audio_processed_s}")
                        print(f"current_block_commit_point_s: {current_block_commit_point_s}")
                        print(f"current_block_commit_point_samples: {current_block_commit_point_samples}")
                        print(f"unprocessed_audio_buffer: {unprocessed_audio_buffer}")
                        print(f"len(raw_audio_block): {len(raw_audio_block)}")
                        print(f"len(segments): {len(segments)}")
                        
                        break

            ffmpeg_process.wait()
            print(f"\n\n{'='*50}")
            print(f"Transcription complete!")
            print(f"SRT file saved to: {srt_path}")
            total_runtime = time.time() - transcription_start_time
            print(f"Total processing time: {format_duration(total_runtime)}")
            print(f"{'='*50}")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            raise e

if __name__ == "__main__":
    transcriber = AudioStreamTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8"
    )
    try:
        input_path = input("Please enter the full path to your audio file: ").strip().strip('"')
        if not os.path.isfile(input_path):
            print(f"Error: The file '{input_path}' was not found.", file=sys.stderr)
            sys.exit(1)

        directory, filename = os.path.split(input_path)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{base_name}.srt")
        
        transcriber.transcribe_to_srt(input_path, output_path)
    except KeyboardInterrupt:
        print("\n\nTranscription cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during setup: {e}")
        raise e