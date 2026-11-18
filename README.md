# Pixel Pit Crew

F1 sponsor brand detection web app. Upload race footage and get per-frame detection results, brand share analytics, and an annotated output video.

---

## How it works

Each video is run through a YOLOv8 model trained on F1 sponsor logos. Every frame goes through inference, bounding boxes are drawn on the annotated output, and detection data is aggregated into brand-level stats.

Pipeline overview:

1. **Upload**: the video is saved to a temp file and a background job is started.
2. **Original re-encode**: ffmpeg starts re-encoding the original to H.264 straight away, so it runs in parallel with inference. This clean copy powers the annotation toggle.
3. **Chunked inference**: the video is split into 2000-frame chunks, which are processed in parallel threads. Each thread has its own YOLO model instance (and its own CUDA stream on NVIDIA GPUs), and frames are batched 32 at a time per model call. Half precision is used on CUDA and MPS.
4. **Stitch & encode**: annotated chunks are concatenated, the original audio is muxed back in, and the result is encoded to H.264 for browser playback.

### Hardware acceleration

The backend picks the best available device and codec automatically:

| | Inference | Video encode | Video decode |
|---|---|---|---|
| **NVIDIA GPU** (Windows/Linux) | CUDA (FP16) | NVENC (`h264_nvenc`) | NVDEC (`-hwaccel cuda`) |
| **Apple Silicon** | MPS (FP16) | VideoToolbox | VideoToolbox |
| **CPU only** | CPU | `libx264` (veryfast) | software |

The number of parallel chunk workers is based on free VRAM on CUDA (~450 MB per model instance), capped at 2 on MPS, and equal to the CPU count otherwise.

---

## Setup

**Requirements:** Python 3.10+ and **ffmpeg on your PATH**. The server will refuse to start without ffmpeg.

### 1. Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Windows
winget install ffmpeg
```

Or download it from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH. Restart your terminal afterwards and check that `ffmpeg -version` works.

### 2. Clone and create a virtual environment

```bash
git clone git@github.com:Swas26/DLCNN_Ass_3.git
cd DLCNN_Ass_3

python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

`requirements.txt` pins a **CUDA 12.8** build of PyTorch (`torch==2.11.0+cu128`). Install it from the PyTorch index:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**Apple Silicon / CPU-only machines:** the `+cu128` builds are not available there. Change the `torch` and `torchvision` lines in `requirements.txt` to plain versions (`torch==2.11.0`, `torchvision==0.26.0`) and run `pip install -r requirements.txt`. MPS is used automatically on Apple Silicon.

**Other CUDA versions:** pick the matching build from [pytorch.org](https://pytorch.org/get-started/locally/) and adjust the index URL and version suffix to match.

To confirm the GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Running

```bash
python backend.py
```

This starts Flask on `http://0.0.0.0:5001`. Open `http://localhost:5001` in a browser; the frontend is served automatically.

---

## Usage

1. Drag a video onto the upload zone or click **Upload Video** (MP4, MOV, AVI, WebM, OGV; max 500 MB).
2. Set **Min confidence** with the slider (0.10–0.90, default 0.40). Detections below this threshold are discarded.
3. Wait for processing. A progress bar is shown for each chunk, followed by a transcoding stage.
4. The results screen shows:
   - The annotated video player, with an **Annotations ON/OFF** toggle
   - A detection table with brand, occurrences, confidence, and a peak-frame button that seeks to the brand's highest-confidence frame
   - A **Brand Share** bar chart by detection count
5. Click **Export CSV** to download the full per-frame data, or **Reupload** to start again.

---

## REST API

### `POST /api/detect`

Upload a video for processing.

**Form fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `video` | file | required | Video file (`.mp4`, `.mov`, `.avi`, `.webm`, `.ogv`, `.ogg`) |
| `conf` | float | `0.40` | Minimum confidence threshold (clamped to 0.01–0.99) |
| `skip` | int | `1` | Run inference on every Nth frame (clamped to 1–30). Skipped frames are still written to the output video, without boxes. |

**Response:**
```json
{ "job_id": "uuid-string" }
```

Errors: `400` if no file is given, `415` for an unsupported extension.

---

### `GET /api/progress/<job_id>`

Poll for job status. The job is removed from memory once a `done` or `error` response has been returned, so later polls return `404`.

**While processing:**
```json
{
  "status": "processing",
  "current": 840,
  "total": 3600,
  "n_chunks": 2,
  "chunk_done": [840, 0],
  "chunk_total": [2000, 1600],
  "transcoding": false
}
```

`transcoding` becomes `true` once inference has finished and the chunks are being stitched and encoded.

**On completion:**
```json
{
  "status": "done",
  "result": {
    "video_url": "/api/video/<job_id>",
    "original_url": "/api/original/<job_id>",
    "duration": 120.0,
    "fps": 30.0,
    "n_chunks": 2,
    "conf": 0.4,
    "brands": ["Ferrari", "Mercedes", "RedBull", "..."],
    "frames": [
      {
        "frame": 0,
        "second": 0.0,
        "detections": [
          {
            "brand": "RedBull",
            "confidence": 0.872,
            "bbox": { "x1": 0.12, "y1": 0.45, "x2": 0.28, "y2": 0.61 }
          }
        ]
      }
    ]
  }
}
```

- `brands` lists every class the model knows (sorted), not only those detected.
- Bounding box coordinates are normalised (0–1, relative to frame size).
- `original_url` is `null` if re-encoding the original failed.

**On error** (HTTP 500):
```json
{ "status": "error", "error": "description" }
```

---

### `GET /api/video/<job_id>`

Streams the annotated output video. Supports HTTP range requests for seeking.

### `GET /api/original/<job_id>`

Streams the original video re-encoded to H.264 (used by the annotation toggle). Returns `404` if the re-encode failed.

---

## Project structure

```
.
├── backend.py          # Flask server, inference + ffmpeg pipeline
├── requirements.txt
├── model/
│   └── best.pt         # Trained YOLOv8 weights
└── static/             # Frontend (served by Flask)
    ├── index.html
    ├── app.js
    ├── style.css
    └── UTSLogoFullVersionPrimaryRGBBLK.jpg
```

---

## Configuration

These constants at the top of `backend.py` control performance:

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_FRAMES` | `2000` | Frames per parallel processing chunk |
| `BATCH_SIZE` | `32` | Frames batched per model call |

A larger `BATCH_SIZE` improves GPU/MPS throughput but uses more memory; lower it if you hit out-of-memory errors. A smaller `CHUNK_FRAMES` creates more parallel chunks but adds overhead (each worker loads its own model).

The upload limit (500 MB) is set via `app.config['MAX_CONTENT_LENGTH']`.

---

## Troubleshooting

- **`ERROR: ffmpeg is not installed or not on your PATH`**: install ffmpeg (see Setup) and restart your terminal.
- **Running on CPU despite having an NVIDIA GPU**: you probably installed a CPU-only torch build. Reinstall with the `--extra-index-url` for your CUDA version.
- **CUDA out of memory**: lower `BATCH_SIZE`.
- **Annotation toggle missing**: re-encoding the original video failed; check the server console for ffmpeg errors.

---

## Status

Prototype, so expect rough edges. Jobs and videos are kept in memory and temp files, so they are lost when the server restarts. The model (`best.pt`) was trained on a limited set of F1 sponsor logos; detection quality depends on footage resolution, camera angle, and whether a brand appeared in the training data.
