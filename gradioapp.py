import torch
from transformers import T5EncoderModel
from diffusers import LTXPipeline, GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
import gradio as gr
import numpy as np
import tempfile
from PIL import Image

model_path = (
    "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltx-video-2b-v0.9-q8_0.gguf"
    )
transformer = LTXVideoTransformer3DModel.from_single_file(
    model_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    )

text_encoder = T5EncoderModel.from_pretrained(
    "calcuis/ltxv-gguf",
    gguf_file="t5xxl_fp16-q4_0.gguf",
    torch_dtype=torch.bfloat16,
    )

pipe = LTXPipeline.from_pretrained(
    "callgg/ltxv-decoder",
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=torch.bfloat16
    ).to("cuda")

prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

def generate(prompt, negative_prompt, input_image_filepath, input_video_filepath,
             height_ui, width_ui, mode,
             duration_ui, ui_frames_to_use,
             seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag,
             progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed_ui = np.random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(int(seed_ui))
    num_frames = max(1, int(duration_ui * 12))  # 12 fps for short demo, adjust as needed
    # Clamp num_frames to a reasonable range
    num_frames = min(max(num_frames, 9), 65)
    
    if mode == "image-to-video" and input_image_filepath:
        # Load image and convert to PIL.Image if needed
        if isinstance(input_image_filepath, str):
            image = Image.open(input_image_filepath).convert("RGB")
        else:
            image = input_image_filepath
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=int(width_ui),
            height=int(height_ui),
            num_frames=num_frames,
            num_inference_steps=50,
            guidance_scale=float(ui_guidance_scale),
            generator=generator,
        ).frames[0]
    else:
        # Default: text-to-video
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width_ui),
            height=int(height_ui),
            num_frames=num_frames,
            num_inference_steps=50,
            guidance_scale=float(ui_guidance_scale),
            generator=generator,
        ).frames[0]
    temp_dir = tempfile.mkdtemp()
    output_video_path = f"{temp_dir}/output.mp4"
    export_to_video(video, output_video_path, fps=24)
    return output_video_path, seed_ui

# --- Gradio UI Definition (copied from app.py, adapted for this pipeline) ---
css="""
#col-container {
    margin: 0 auto;
    max-width: 900px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# LTX Video 0.9.7 Distilled")
    gr.Markdown("Fast high quality video generation. [Model](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors) [GitHub](https://github.com/Lightricks/LTX-Video) [Diffusers](#)")
    
    with gr.Row():
        with gr.Column():
            with gr.Tab("image-to-video") as image_tab:
                video_i_hidden = gr.Textbox(label="video_i", visible=False, value=None)
                image_i2v = gr.Image(label="Input Image", type="filepath", sources=["upload", "webcam", "clipboard"])
                i2v_prompt = gr.Textbox(label="Prompt", value="The creature from the image starts to move", lines=3)
                i2v_button = gr.Button("Generate Image-to-Video", variant="primary")
            with gr.Tab("text-to-video") as text_tab:
                image_n_hidden = gr.Textbox(label="image_n", visible=False, value=None)
                video_n_hidden = gr.Textbox(label="video_n", visible=False, value=None)
                t2v_prompt = gr.Textbox(label="Prompt", value="A majestic dragon flying over a medieval castle", lines=3)
                t2v_button = gr.Button("Generate Text-to-Video", variant="primary")
            with gr.Tab("video-to-video", visible=False) as video_tab:
                image_v_hidden = gr.Textbox(label="image_v", visible=False, value=None)
                video_v2v = gr.Video(label="Input Video", sources=["upload", "webcam"])
                frames_to_use = gr.Slider(label="Frames to use from input video", minimum=9, maximum=25, value=9, step=8, info="Number of initial frames to use for conditioning/transformation. Must be N*8+1.")
                v2v_prompt = gr.Textbox(label="Prompt", value="Change the style to cinematic anime", lines=3)
                v2v_button = gr.Button("Generate Video-to-Video", variant="primary")

            duration_input = gr.Slider(
                label="Video Duration (seconds)", 
                minimum=0.3, 
                maximum=8.5, 
                value=2,  
                step=0.1, 
                info=f"Target video duration (0.3s to 8.5s)"
            )
            improve_texture = gr.Checkbox(label="Improve Texture (multi-scale)", value=True, info="Uses a two-pass generation for better quality, but is slower. Recommended for final output.")

        with gr.Column():
            output_video = gr.Video(label="Generated Video", interactive=False)

    with gr.Accordion("Advanced settings", open=False):
        mode = gr.Dropdown(["text-to-video", "image-to-video", "video-to-video"], label="task", value="image-to-video", visible=False)
        negative_prompt_input = gr.Textbox(label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", lines=2)
        with gr.Row():
            seed_input = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=2**32-1)
            randomize_seed_input = gr.Checkbox(label="Randomize Seed", value=True)
        with gr.Row():
            guidance_scale_input = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=10.0, value=1.0, step=0.1, info="Controls how much the prompt influences the output. Higher values = stronger influence.")
        with gr.Row():
            height_input = gr.Slider(label="Height", value=480, step=32, minimum=256, maximum=1024, info="Must be divisible by 32.")
            width_input = gr.Slider(label="Width", value=704, step=32, minimum=256, maximum=1280, info="Must be divisible by 32.")

    t2v_inputs = [t2v_prompt, negative_prompt_input, image_n_hidden, video_n_hidden,
                  height_input, width_input, mode,
                  duration_input, frames_to_use, 
                  seed_input, randomize_seed_input, guidance_scale_input, improve_texture]
    i2v_inputs = [i2v_prompt, negative_prompt_input, image_i2v, video_i_hidden,
                  height_input, width_input, mode,
                  duration_input, frames_to_use, 
                  seed_input, randomize_seed_input, guidance_scale_input, improve_texture]
    v2v_inputs = [v2v_prompt, negative_prompt_input, image_v_hidden, video_v2v,
                  height_input, width_input, mode,
                  duration_input, frames_to_use, 
                  seed_input, randomize_seed_input, guidance_scale_input, improve_texture]

    t2v_button.click(fn=generate, inputs=t2v_inputs, outputs=[output_video, seed_input], api_name="text_to_video")
    i2v_button.click(fn=generate, inputs=i2v_inputs, outputs=[output_video, seed_input], api_name="image_to_video")
    v2v_button.click(fn=generate, inputs=v2v_inputs, outputs=[output_video, seed_input], api_name="video_to_video")

if __name__ == "__main__":
    demo.queue().launch(server_name='0.0.0.0', server_port=7860, debug=True)
