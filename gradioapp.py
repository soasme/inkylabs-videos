import torch
from transformers import T5EncoderModel
from diffusers import LTXImageToVideoPipeline, GGUFQuantizationConfig, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
import gradio as gr
import numpy as np
import tempfile
from PIL import Image

model_path = (
    "https://huggingface.co/calcuis/ltxv-gguf/blob/main/ltxv-2b-0.9.6-distilled-fp32-q8_0.gguf"
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

pipe = LTXImageToVideoPipeline.from_pretrained(
    "callgg/ltxv-decoder",
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=torch.bfloat16
    ).to("cuda")

def generate(prompt, negative_prompt, input_image_filepath,
             height_ui, width_ui,
             duration_ui,
             seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag,
             progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed_ui = np.random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(int(seed_ui))
    num_frames = max(1, int(duration_ui * 12))
    num_frames = min(max(num_frames, 9), 65)
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
        num_inference_steps=100,
        guidance_scale=float(ui_guidance_scale),
        generator=generator,
    ).frames[0]
    temp_dir = tempfile.mkdtemp()
    output_video_path = f"{temp_dir}/output.mp4"
    export_to_video(video, output_video_path, fps=24)
    return output_video_path, seed_ui

css="""
#col-container {
    margin: 0 auto;
    max-width: 900px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# LTX Video 0.9.7 Distilled (Image-to-Video Only)")
    gr.Markdown("Fast high quality image-to-video generation. [Model](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors) [GitHub](https://github.com/Lightricks/LTX-Video) [Diffusers](#)")
    
    with gr.Row():
        with gr.Column():
            image_i2v = gr.Image(label="Input Image", type="filepath", sources=["upload", "webcam", "clipboard"])
            i2v_prompt = gr.Textbox(label="Prompt", value="The creature from the image starts to move", lines=3)
            i2v_button = gr.Button("Generate Image-to-Video", variant="primary")
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
        negative_prompt_input = gr.Textbox(label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", lines=2)
        with gr.Row():
            seed_input = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=2**32-1)
            randomize_seed_input = gr.Checkbox(label="Randomize Seed", value=True)
        with gr.Row():
            guidance_scale_input = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=10.0, value=1.0, step=0.1, info="Controls how much the prompt influences the output. Higher values = stronger influence.")
        with gr.Row():
            height_input = gr.Slider(label="Height", value=480, step=32, minimum=256, maximum=1024, info="Must be divisible by 32.")
            width_input = gr.Slider(label="Width", value=704, step=32, minimum=256, maximum=1280, info="Must be divisible by 32.")

    i2v_inputs = [i2v_prompt, negative_prompt_input, image_i2v,
                  height_input, width_input,
                  duration_input,
                  seed_input, randomize_seed_input, guidance_scale_input, improve_texture]
    i2v_button.click(fn=generate, inputs=i2v_inputs, outputs=[output_video, seed_input], api_name="image_to_video")

if __name__ == "__main__":
    demo.queue().launch(server_name='0.0.0.0', server_port=7860, debug=True)
