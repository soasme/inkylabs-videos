# inkylabs-videos

## Getting Started

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

## VRAM Requirement

This application loads multiple large models into GPU memory (VRAM) simultaneously, including the main video generation model and a spatial upscaler. As a result, VRAM requirements are higher than for single-model inference.

- **Minimum:** 16 GB VRAM (may work with reduced settings, but not guaranteed)
- **Recommended:** 24 GB VRAM or higher (for stable operation, larger resolutions, and multi-scale upscaling)

Actual requirements depend on the selected model, video resolution, and whether multi-scale (texture improvement) is enabled. For best results, use a GPU with at least 24 GB VRAM.