import gradio as gr
import torch
from transformers import T5EncoderModel
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, GGUFQuantizationConfig
from diffusers.utils import export_to_video
import os
import gc # For garbage collection to manage VRAM

# --- Model Loading (Global to avoid reloading on each call) ---
# This part is crucial and should ideally happen once when the app starts.
# We'll put it in a function that can be called conditionally or globally.

pipe = None # Initialize pipe as None

def load_model(quantization_config_type="q8_0", torch_dtype="bfloat16"):
    global pipe
    if pipe is not None:
        print("Model already loaded.")
        return

    print(f"Loading LTX-Video model with quantization: {quantization_config_type}, dtype: {torch_dtype}")

    # Determine compute_dtype based on torch_dtype
    compute_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

    # Define model paths (local or Hugging Face)
    # Using local paths is often more stable and faster for repeated runs
    main_model_path = "ltx-video-2b-v0.9-q8_0.gguf" # Assume this is in the same directory
    text_encoder_gguf_file = "t5xxl_fp16-q4_0.gguf" # Assume this is in the same directory

    # --- Attempt to download if not local ---
    # You might want to add more robust download logic here
    # For now, let's assume they are either local or directly accessible by diffusers
    
    # NOTE: LTXVideoTransformer3DModel.from_single_file expects a local path or a blob URL
    # It's better to download the GGUF files first if they are not already present.
    # We will provide a simple download mechanism for this example.
    
    try:
        # Check if main model file exists
        if not os.path.exists(main_model_path):
            print(f"Downloading {main_model_path} from Hugging Face...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="calcuis/ltxv-gguf", filename="ltx-video-2b-v0.9-q8_0.gguf", local_dir=".")
            print(f"Downloaded {main_model_path}.")
        
        # Check if text encoder file exists
        if not os.path.exists(text_encoder_gguf_file):
            print(f"Downloading {text_encoder_gguf_file} from Hugging Face...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="calcuis/ltxv-gguf", filename="t5xxl_fp16-q4_0.gguf", local_dir=".")
            print(f"Downloaded {text_encoder_gguf_file}.")

        # Transformer
        print("Loading Transformer...")
        transformer_quant_config = GGUFQuantizationConfig(
            compute_dtype=compute_dtype,
        )
        transformer = LTXVideoTransformer3DModel.from_single_file(
            main_model_path,
            quantization_config=transformer_quant_config,
            torch_dtype=compute_dtype, # Use compute_dtype for model weights precision
        )
        print("Transformer loaded.")

        # Text Encoder
        print("Loading Text Encoder...")
        text_encoder = T5EncoderModel.from_pretrained(
            "calcuis/ltxv-gguf", # This will look for t5xxl_fp16-q4_0.gguf in the hub or local cache
            gguf_file=text_encoder_gguf_file,
            torch_dtype=compute_dtype,
        )
        print("Text Encoder loaded.")

        # Pipeline
        print("Loading Pipeline...")
        pipe = LTXPipeline.from_pretrained(
            "callgg/ltxv-decoder", # This is the diffusion decoder part
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=compute_dtype
        )
        
        # Move to CUDA if available, otherwise stay on CPU
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("Pipeline moved to CUDA.")
        else:
            print("CUDA not available, pipeline loaded to CPU.")
            # For GGUF, CPU inference is often the default if GPU isn't specified,
            # but performance will be significantly slower.
            # You might need to manually set device for transformer and text_encoder if they don't move with pipe.
            transformer.to("cpu") 
            text_encoder.to("cpu")
            pipe.to("cpu")
            
        print("LTX-Video model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        gr.Warning(f"Error loading model. Please check console for details: {e}")
        pipe = None # Reset pipe if loading fails
        # Attempt to clear memory if loading failed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e # Re-raise to show in Gradio as well


# --- Video Generation Function ---
def generate_video(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int
) -> str: # Returns path to the generated video file
    
    if pipe is None:
        gr.Error("Model not loaded. Please ensure model files are available and try again.")
        return None

    gr.Info("Starting video generation... This may take a while.")
    print(f"--- Generating Video ---")
    print(f"Prompt: {prompt}")
    print(f"Negative Prompt: {negative_prompt}")
    print(f"Resolution: {width}x{height}")
    print(f"Number of Frames: {num_frames}")
    print(f"Inference Steps: {num_inference_steps}")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Seed: {seed}")

    try:
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 and torch.cuda.is_available() else torch.Generator("cpu").manual_seed(seed) if seed != -1 else None

        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np" # Get numpy array of frames
        ).frames[0] # Assuming frames are returned directly or as a list in .frames[0]
        
        output_filename = f"output_{int(time.time())}.mp4"
        output_dir = "generated_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        print(f"Exporting video to {output_path}...")
        # LTXPipeline's output_type="np" might give numpy array.
        # Ensure 'export_to_video' gets a list of numpy frames.
        # If video_frames is already a list of arrays, no change needed.
        # If it's a single numpy array representing all frames, it might need reshaping.
        # Based on diffusers example: pipe(...).frames[0] implies a list of numpy arrays.
        export_to_video(video_frames, output_path, fps=24) # Default FPS is 24

        gr.Info("Video generation complete!")
        # Clear VRAM after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return output_path

    except torch.cuda.OutOfMemoryError as e:
        gr.Error(f"CUDA Out Of Memory! Please try reducing resolution, number of frames, or using a lower quantization model. Error: {e}")
        print(f"CUDA OOM Error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as e:
        gr.Error(f"An error occurred during video generation: {e}")
        print(f"Error during generation: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None


# --- Gradio Interface Definition ---

# Define dropdown choices
resolutions = ["512x288", "704x480", "768x432", "1024x576"] # LTX-Video specific common resolutions
inference_steps_options = [25, 50, 75, 100]

with gr.Blocks(title="LTX-Video GGUF Generator") as demo:
    gr.Markdown(
        """
        # LTX-Video GGUF Generator
        Generate videos using the LTX-Video model with GGUF files via the `diffusers` library.
        
        **VRAM Warning:** Video generation is memory intensive. Ensure you have sufficient GPU VRAM (8GB+ recommended, 12GB+ for smoother experience). If you encounter CUDA Out of Memory errors, try reducing resolution or number of frames.
        
        **Model Loading:** The model will be loaded the first time you interact with the app. This might take a moment and download files if not present locally.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A woman with long brown hair and light skin smiles at another woman with long blonde hair. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage",
                lines=3
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="worst quality, inconsistent motion, blurry, jittery, distorted",
                lines=2
            )
            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=704,
                    interactive=True
                )
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=768,
                    step=64,
                    value=480,
                    interactive=True
                )
            with gr.Row():
                num_frames = gr.Slider(
                    label="Number of Frames",
                    minimum=16,
                    maximum=128,
                    step=8,
                    value=25,
                    interactive=True
                )
                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=10,
                    maximum=100,
                    step=5,
                    value=50,
                    interactive=True
                )
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=7.0,
                    interactive=True
                )
                seed = gr.Slider(
                    label="Seed (-1 for random)",
                    minimum=-1,
                    maximum=999999999,
                    step=1,
                    value=42,
                    interactive=True
                )
            
            # Placeholder for model loading and selection (optional, can be expanded)
            # For this specific example, we hardcode the model and dtype as per your code.
            # You could add dropdowns for different GGUF files and dtypes if you have them.
            
            gr.Markdown(
                """
                **Model Details (Hardcoded in this demo):**
                - Main Model: `ltx-video-2b-v0.9-q8_0.gguf`
                - Text Encoder: `t5xxl_fp16-q4_0.gguf`
                - Compute Dtype: `torch.bfloat16`
                """
            )

            generate_btn = gr.Button("Generate Video", variant="primary")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video", interactive=False)
            gr.Examples(
                examples=[
                    ["A futuristic cityscape at sunset, with flying cars.", "blurry, low quality", 704, 480, 25, 50, 7.0, 123],
                    ["An astronaut floating in space looking at Earth, vibrant colors.", "ugly, distorted", 704, 480, 32, 60, 8.0, 456],
                    ["A serene forest with a gentle waterfall, sunlight filtering through the trees.", "", 512, 288, 25, 50, 7.5, 789],
                ],
                inputs=[prompt, negative_prompt, width, height, num_frames, num_inference_steps, guidance_scale, seed]
            )

    # --- Load Model on app startup (or first interaction) ---
    # Using a gr.on_load event is a good way to handle this
    demo.load(load_model) # This will call load_model when the app is loaded

    # --- Event Listener for Generation ---
    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt,
            negative_prompt,
            width,
            height,
            num_frames,
            num_inference_steps,
            guidance_scale,
            seed
        ],
        outputs=output_video,
        api_name="generate_video" # Allows calling via Gradio API if needed
    )

# Launch the app
if __name__ == "__main__":
    demo.queue() # Enable queuing for multiple requests
    demo.launch(server_name='0.0.0.0', server_port=7860, debug=True)
