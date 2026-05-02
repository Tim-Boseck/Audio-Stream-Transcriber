# Audio Stream Transcriber

A private utility for high-performance, memory-efficient audio transcription. While originally designed to transcribe long-form audiobooks, this tool has evolved into a robust engine for transcribing long-duration recordings of meetings, lectures, and conversations.

## 🛠️ Project Evolution
This project has undergone significant architectural shifts to handle increasingly complex audio profiles:

* **v1–v2 (The In-Memory Era):** Used `pydub` to load entire files into RAM. Worked well for short clips but caused system instability with large audiobooks.
* **v3–v5 (The Streaming Era):** Introduced FFmpeg and `soundfile`. Switched to a pipe-based architecture, allowing the tool to process files of any length by streaming chunks from disk without exhausting memory.
* **v6 (The Optimization Era):** Implemented **Active Silence Culling**. Added logic to detect and "prune" the audio buffer during periods of silence/inactivity. This prevents the "exponential slowdown" bug where the engine would otherwise accumulate and re-transcribe mounting amounts of silent data.

## 🚀 Key Features

* **True Stream Processing:** Leverages FFmpeg `stdout` pipes to feed raw PCM data directly into the Whisper model, ensuring constant memory usage regardless of file size.
* **Advanced Buffer Management:** Intelligent slicing of the audio buffer during silence to prevent processing bottlenecks and infinite loops.
_   **Real-Time Dashboard:** A live CLI interface providing:
    *   Total runtime.
    *   Progress tracking (Committed time vs. Total duration).
    *   Live transcription snippets as they are committed to disk.
    *   Dynamic ETA calculation based on current processing speed.
* **Format Agnostic:** Uses FFmpeg as the backend, allowing for the transcription of MP3, WAV, M4A, FLAC, and nearly any other audio format.

## ⚙️ Setup & Requirements

### Prerequisites
1.  **Python 3.8+**
2.  **FFmpeg:** Essential for the streaming pipeline.
    *   **Windows:** `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/).
    *   **macOS:** `brew install ffmpeg`
    *   **Linux:** `sudo apt install ffmpeg`

### Installation
```bash
pip install faster-whisper numpy
```

## 📖 Usage
Run the script via your terminal:
```bash
python audiobook6.py
```
The script will prompt you for the full path to your audio file. It will then automatically determine the output `.srt` path in the same directory.

## 🗺️ Future Roadmap

* [ ] **CLI Argument Parsing:** Implement `argparse` to allow passing file paths and settings (like `--model-size`) directly via the command line.
* [ ] **Resume/Continue Capability:** Add a feature to parse existing `.srt` files to find the last timestamped segment, allowing the transcriber to skip already-processed audio and resume interrupted jobs.
* [ ] **Hardware Auto-Detection:** Implement logic to automatically detect NVIDIA GPUs and configure `device="cuda"` and `compute_type="float16"` without manual code edits.
* [ ] **Batch Processing:** Implement a queue system or `ProcessPoolExecutor` to handle directories of files or multiple simultaneous transcription jobs.

## ⚖️ License
**Private Project.** This is a personal utility for individual use. No license is provided, and redistribution or use of this code is not intended.
