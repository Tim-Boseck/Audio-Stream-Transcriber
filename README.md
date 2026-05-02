# Audio Stream Transcriber (v0.6)

A personal workflow utility designed to leverage the high-performance capabilities of **faster-whisper** for long-duration audio transcription. 

> **Note on Core Technology:** This project is a convenience wrapper and does not contain any Whisper models or inference logic itself. It relies entirely on [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper), a highly efficient reimplementation of [openai/whisper](https://github.com/openai/whisper) model using [CTranslate2](https://github.com/OpenNMT/CTranslate2).

## 🎯 Use Cases
This tool was developed as a personal convenience to manage two primary workflows:
1.  **Audiobook Transcription:** Handling massive, multi-hour audio files that would otherwise exhaust system RAM.
2.  **Meeting & Lecture Archiving:** Processing long-duration recordings of conversations, classes, or meetings where the ability to extract clean, timestamped text (and subsequently plain text) is essential.

## 🚀 Key Features

### `audio-stream-transcriber.py` (The Engine)
*   **Streaming Architecture:** Uses an FFmpeg-to-Python pipe to stream raw PCM data directly from disk. This ensures that memory usage remains constant, regardless of whether you are transcribing a 5-minute clip or a 20-hour recording.
*   **Active Silence Culling:** Implements intelligent buffer management to detect and "prune" the audio buffer during periods of silence. This prevents the exponential slowdowns (buffer bloat) that occur when a transcription engine attempts to re-process growing amounts of silent data.
*   **Real-Time Telemetry:** A live CLI dashboard providing:
    *   Cumulative runtime tracking.
    *   Progress metrics (Committed audio vs. Total duration).
    *   Dynamic ETA calculation based on real-time processing throughput.
*   **Robustness:** Designed to handle the "end-of-file" edge cases, ensuring that any residual data left in the buffer is processed and committed before the process terminates.

### `minimize_srt_to_raw.py` (The Post-Processor)
*   **Text Extraction:** A utility to strip SRT timecode metadata and convert subtitles into clean, human-readable plain text.
*   / **De-duplication:** Automatically removes consecutive duplicate lines to ensure the resulting `.txt` file is a clean transcript rather than a repetitive log of timestamps.

## 🛠️ Requirements & Setup

### Prerequisites
*   **Python 3.8+**
*   **FFmpeg:** Essential for the streaming pipeline and audio decoding.
    *   **Windows:** `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/).
    *   **macOS:** `brew install ffmpeg`
    *   **Linux:** `sudo apt install ffmpeg`

### Installation
```bash
# Install the required dependencies via pip
pip install -r requirements.txt
```

## 🗺️ Future Roadmap
* [ ] **CLI Argument Parsing:** Move away from interactive prompts to a standard `argparse` interface for easier automation.
* [ ] **Resume/Continue Capability:** Implement logic to parse existing `.srt` files to skip already-processed audio segments and resume interrupted jobs.
* [ ] **Hardware Auto-Detection:** Add automatic detection of NVIDIA GPUs to configure `device="cuda"` and `compute_type="float16"` without manual script modification.
* [ ] **Batch Processing:** Support for directory-wide transcription via a process queue.

## ⚖️ License
**Private Project.** This is a personal utility for individual use. No license is provided, and redistribution or use of this code is not intended.