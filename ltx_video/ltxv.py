from mmgp import offload
import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional, List, Union
import yaml

import imageio
import json
import numpy as np
import torch
from safetensors import safe_open
from PIL import Image
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from huggingface_hub import hf_hub_download

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.pipelines import crf_compressor
from ltx_video.utils.utils import calculate_new_dimensions
import cv2

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

logger = logging.get_logger("LTX-Video")


def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_memory
    return 0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image_to_tensor_with_resize_and_crop(
    image_input: Union[str, Image.Image],
    target_height: int = 512,
    target_width: int = 768,
    just_crop: bool = False,
) -> torch.Tensor:
    """Load and process an image into a tensor.

    Args:
        image_input: Either a file path (str) or a PIL Image object
        target_height: Desired height of output tensor
        target_width: Desired width of output tensor
        just_crop: If True, only crop the image to the target size without resizing
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")

    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    if not just_crop:
        image = image.resize((target_width, target_height))

    image = np.array(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    frame_tensor = torch.from_numpy(image).float()
    frame_tensor = crf_compressor.compress(frame_tensor / 255.0) * 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)



def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding




def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class LTXV:

    def __init__(
        self,
        model_filepath: str,
        text_encoder_filepath: str,
        dtype = torch.bfloat16,
        VAE_dtype = torch.bfloat16, 
        mixed_precision_transformer = False
    ):

        # if dtype == torch.float16:
        dtype  = torch.bfloat16
        self.mixed_precision_transformer = mixed_precision_transformer
        self.distilled = any("lora" in name for name in model_filepath)
        model_filepath = [name for name in model_filepath if not "lora" in name ]
        # with safe_open(ckpt_path, framework="pt") as f:
        #     metadata = f.metadata()
        #     config_str = metadata.get("config")
        #     configs = json.loads(config_str)
        #     allowed_inference_steps = configs.get("allowed_inference_steps", None)
        # transformer = Transformer3DModel.from_pretrained(ckpt_path)
        # transformer = offload.fast_load_transformers_model("c:/temp/ltxdistilled/diffusion_pytorch_model-00001-of-00006.safetensors",  forcedConfigPath="c:/temp/ltxdistilled/config.json")

        # vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
        vae = offload.fast_load_transformers_model("ckpts/ltxv_0.9.7_VAE.safetensors", modelClass=CausalVideoAutoencoder)
        # if VAE_dtype == torch.float16:
        VAE_dtype = torch.bfloat16

        vae = vae.to(VAE_dtype)
        vae._model_dtype = VAE_dtype
        # vae = offload.fast_load_transformers_model("vae.safetensors", modelClass=CausalVideoAutoencoder, modelPrefix= "vae",  forcedConfigPath="config_vae.json")
        # offload.save_model(vae, "vae.safetensors", config_file_path="config_vae.json")

        # model_filepath = "c:/temp/ltxd/ltxv-13b-0.9.7-distilled.safetensors"
        transformer = offload.fast_load_transformers_model(model_filepath, modelClass=Transformer3DModel) 
        # offload.save_model(transformer, "ckpts/ltxv_0.9.7_13B_distilled_bf16.safetensors", config_file_path= "c:/temp/ltxd/config.json")
        # offload.save_model(transformer, "ckpts/ltxv_0.9.7_13B_distilled_quanto_bf16_int8.safetensors", do_quantize= True, config_file_path="c:/temp/ltxd/config.json")
        # transformer = offload.fast_load_transformers_model(model_filepath, modelClass=Transformer3DModel) 
        transformer._model_dtype = dtype
        if mixed_precision_transformer:
            transformer._lock_dtype = torch.float


        scheduler = RectifiedFlowScheduler.from_pretrained("ckpts/ltxv_scheduler.json")
        # transformer = offload.fast_load_transformers_model("ltx_13B_quanto_bf16_int8.safetensors", modelClass=Transformer3DModel, modelPrefix= "model.diffusion_model",  forcedConfigPath="config_transformer.json")
        # offload.save_model(transformer, "ltx_13B_quanto_bf16_int8.safetensors", do_quantize= True, config_file_path="config_transformer.json")

        latent_upsampler = LatentUpsampler.from_pretrained("ckpts/ltxv_0.9.7_spatial_upscaler.safetensors").to("cpu").eval()
        latent_upsampler.to(VAE_dtype)
        latent_upsampler._model_dtype = VAE_dtype

        allowed_inference_steps = None

        # text_encoder = T5EncoderModel.from_pretrained(
        #     "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
        # )
        # text_encoder.to(torch.bfloat16)
        # offload.save_model(text_encoder, "T5_xxl_1.1_enc_bf16.safetensors", config_file_path="T5_config.json")
        # offload.save_model(text_encoder, "T5_xxl_1.1_enc_quanto_bf16_int8.safetensors", do_quantize= True, config_file_path="T5_config.json")

        text_encoder = offload.fast_load_transformers_model(text_encoder_filepath)
        patchifier = SymmetricPatchifier(patch_size=1)
        tokenizer = T5Tokenizer.from_pretrained( "ckpts/T5_xxl_1.1")

        enhance_prompt = False
        if enhance_prompt:
            prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained( "ckpts/Florence2", trust_remote_code=True)
            prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained( "ckpts/Florence2", trust_remote_code=True)
            prompt_enhancer_llm_model = offload.fast_load_transformers_model("ckpts/Llama3_2_quanto_bf16_int8.safetensors")
            prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained("ckpts/Llama3_2")
        else:
            prompt_enhancer_image_caption_model = None
            prompt_enhancer_image_caption_processor = None
            prompt_enhancer_llm_model = None
            prompt_enhancer_llm_tokenizer = None

        if prompt_enhancer_image_caption_model != None:
            pipe["prompt_enhancer_image_caption_model"] = prompt_enhancer_image_caption_model
            prompt_enhancer_image_caption_model._model_dtype = torch.float
        
            pipe["prompt_enhancer_llm_model"] = prompt_enhancer_llm_model

        # offload.profile(pipe, profile_no=5, extraModelsToQuantize = None, quantizeTransformer = False, budgets = { "prompt_enhancer_llm_model" : 10000, "prompt_enhancer_image_caption_model" : 10000, "vae" : 3000, "*" : 100 }, verboseLevel=2)


        # Use submodels for the pipeline
        submodel_dict = {
            "transformer": transformer,
            "patchifier": patchifier,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "vae": vae,
            "prompt_enhancer_image_caption_model": prompt_enhancer_image_caption_model,
            "prompt_enhancer_image_caption_processor": prompt_enhancer_image_caption_processor,
            "prompt_enhancer_llm_model": prompt_enhancer_llm_model,
            "prompt_enhancer_llm_tokenizer": prompt_enhancer_llm_tokenizer,
            "allowed_inference_steps": allowed_inference_steps,
        }
        pipeline = LTXVideoPipeline(**submodel_dict)
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

        self.pipeline = pipeline
        self.model = transformer 
        self.vae = vae
        # return pipeline, pipe

    def generate(
        self,
        input_prompt: str,
        n_prompt: str,
        image_start = None,
        image_end = None,
        input_video = None,
        sampling_steps = 50,
        image_cond_noise_scale:  float = 0.15,
        input_media_path: Optional[str] = None,
        strength: Optional[float] = 1.0,
        seed: int = 42,
        height: Optional[int] = 704,
        width: Optional[int] = 1216,
        frame_num: int = 81,
        frame_rate: int = 30,
        fit_into_canvas = True,
        callback=None,
        device: Optional[str] = None,
        VAE_tile_size = None,
        **kwargs,
    ):

        num_inference_steps1 = sampling_steps
        num_inference_steps2 = sampling_steps #10
        conditioning_strengths  = None
        conditioning_media_paths = []
        conditioning_start_frames = []


        if input_video != None:
            conditioning_media_paths.append(input_video) 
            conditioning_start_frames.append(0)
            height, width = input_video.shape[-2:]
        else:
            if image_start != None:
                image_start = image_start[0]
                frame_width, frame_height  = image_start.size
                height, width = calculate_new_dimensions(height, width, frame_height, frame_width, fit_into_canvas, 32)
                conditioning_media_paths.append(image_start) 
                conditioning_start_frames.append(0)
            if image_end != None:
                image_end = image_end[0]
                conditioning_media_paths.append(image_end) 
                conditioning_start_frames.append(frame_num-1)

        if len(conditioning_media_paths) == 0:
            conditioning_media_paths = None
            conditioning_start_frames = None

        if self.distilled :
            pipeline_config = "ltx_video/configs/ltxv-13b-0.9.7-distilled.yaml"
        else:
            pipeline_config = "ltx_video/configs/ltxv-13b-0.9.7-dev.yaml"
        # check if pipeline_config is a file
        if not os.path.isfile(pipeline_config):
            raise ValueError(f"Pipeline config file {pipeline_config} does not exist")
        with open(pipeline_config, "r") as f:
            pipeline_config = yaml.safe_load(f)


        # Validate conditioning arguments
        if conditioning_media_paths:
            # Use default strengths of 1.0
            if not conditioning_strengths:
                conditioning_strengths = [1.0] * len(conditioning_media_paths)
            if not conditioning_start_frames:
                raise ValueError(
                    "If `conditioning_media_paths` is provided, "
                    "`conditioning_start_frames` must also be provided"
                )
            if len(conditioning_media_paths) != len(conditioning_strengths) or len(
                conditioning_media_paths
            ) != len(conditioning_start_frames):
                raise ValueError(
                    "`conditioning_media_paths`, `conditioning_strengths`, "
                    "and `conditioning_start_frames` must have the same length"
                )
            if any(s < 0 or s > 1 for s in conditioning_strengths):
                raise ValueError("All conditioning strengths must be between 0 and 1")
            if any(f < 0 or f >= frame_num for f in conditioning_start_frames):
                raise ValueError(
                    f"All conditioning start frames must be between 0 and {frame_num-1}"
                )

        # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((frame_num - 2) // 8 + 1) * 8 + 1

        padding = calculate_padding(height, width, height_padded, width_padded)

        logger.warning(
            f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
        )


        # prompt_enhancement_words_threshold = pipeline_config[
        #     "prompt_enhancement_words_threshold"
        # ]

        # prompt_word_count = len(prompt.split())
        # enhance_prompt = (
        #     prompt_enhancement_words_threshold > 0
        #     and prompt_word_count < prompt_enhancement_words_threshold
        # )

        # # enhance_prompt = False

        # if prompt_enhancement_words_threshold > 0 and not enhance_prompt:
        #     logger.info(
        #         f"Prompt has {prompt_word_count} words, which exceeds the threshold of {prompt_enhancement_words_threshold}. Prompt enhancement disabled."
        #     )


        seed_everething(seed)
        device = device or get_device()
        generator = torch.Generator(device=device).manual_seed(seed)

        media_item = None
        if input_media_path:
            media_item = load_media_file(
                media_path=input_media_path,
                height=height,
                width=width,
                max_frames=num_frames_padded,
                padding=padding,
            )

        conditioning_items = (
            prepare_conditioning(
                conditioning_media_paths=conditioning_media_paths,
                conditioning_strengths=conditioning_strengths,
                conditioning_start_frames=conditioning_start_frames,
                height=height,
                width=width,
                num_frames=frame_num,
                padding=padding,
                pipeline=self.pipeline,
            )
            if conditioning_media_paths
            else None
        )

        stg_mode = pipeline_config.get("stg_mode", "attention_values")
        del pipeline_config["stg_mode"]
        if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
            skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
            skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
            skip_layer_strategy = SkipLayerStrategy.Residual
        elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
            skip_layer_strategy = SkipLayerStrategy.TransformerBlock
        else:
            raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

        # Prepare input for the pipeline
        sample = {
            "prompt": input_prompt,
            "prompt_attention_mask": None,
            "negative_prompt": n_prompt,
            "negative_prompt_attention_mask": None,
        }


        images = self.pipeline(
            **pipeline_config,
            ltxv_model = self,
            num_inference_steps1 = num_inference_steps1,
            num_inference_steps2 = num_inference_steps2,
            skip_layer_strategy=skip_layer_strategy,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=frame_rate,
            **sample,
            media_items=media_item,
            strength=strength,
            conditioning_items=conditioning_items,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=image_cond_noise_scale,
            mixed_precision=pipeline_config.get("mixed", self.mixed_precision_transformer),
            callback=callback,
            VAE_tile_size = VAE_tile_size,
            device=device,
            # enhance_prompt=enhance_prompt,
        )
        if images == None:
            return None

        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom
        pad_right = -pad_right
        if pad_bottom == 0:
            pad_bottom = images.shape[3]
        if pad_right == 0:
            pad_right = images.shape[4]
        images = images[:, :, :frame_num, pad_top:pad_bottom, pad_left:pad_right]
        images = images.sub_(0.5).mul_(2).squeeze(0)
        return images


def prepare_conditioning(
    conditioning_media_paths: List[str],
    conditioning_strengths: List[float],
    conditioning_start_frames: List[int],
    height: int,
    width: int,
    num_frames: int,
    padding: tuple[int, int, int, int],
    pipeline: LTXVideoPipeline,
) -> Optional[List[ConditioningItem]]:
    """Prepare conditioning items based on input media paths and their parameters.

    Args:
        conditioning_media_paths: List of paths to conditioning media (images or videos)
        conditioning_strengths: List of conditioning strengths for each media item
        conditioning_start_frames: List of frame indices where each item should be applied
        height: Height of the output frames
        width: Width of the output frames
        num_frames: Number of frames in the output video
        padding: Padding to apply to the frames
        pipeline: LTXVideoPipeline object used for condition video trimming

    Returns:
        A list of ConditioningItem objects.
    """
    conditioning_items = []
    for path, strength, start_frame in zip(
        conditioning_media_paths, conditioning_strengths, conditioning_start_frames
    ):
        if isinstance(path, Image.Image):
            num_input_frames = orig_num_input_frames =  1
        else:
            num_input_frames = orig_num_input_frames = get_media_num_frames(path)
        if hasattr(pipeline, "trim_conditioning_sequence") and callable(
            getattr(pipeline, "trim_conditioning_sequence")
        ):
            num_input_frames = pipeline.trim_conditioning_sequence(
                start_frame, orig_num_input_frames, num_frames
            )
        if num_input_frames < orig_num_input_frames:
            logger.warning(
                f"Trimming conditioning video {path} from {orig_num_input_frames} to {num_input_frames} frames."
            )

        media_tensor = load_media_file(
            media_path=path,
            height=height,
            width=width,
            max_frames=num_input_frames,
            padding=padding,
            just_crop=True,
        )
        conditioning_items.append(ConditioningItem(media_tensor, start_frame, strength))
    return conditioning_items


def get_media_num_frames(media_path: str) -> int:
    if isinstance(media_path, Image.Image):
        return 1
    elif torch.is_tensor(media_path):
        return media_path.shape[1]
    elif isinstance(media_path, str) and any( media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
        reader = imageio.get_reader(media_path)
        return min(reader.count_frames(), max_frames)
    else:
        raise Exception("video format not supported")


def load_media_file(
    media_path: str,
    height: int,
    width: int,
    max_frames: int,
    padding: tuple[int, int, int, int],
    just_crop: bool = False,
) -> torch.Tensor:
    if isinstance(media_path, Image.Image):
        # Input image
        media_tensor = load_image_to_tensor_with_resize_and_crop(
            media_path, height, width, just_crop=just_crop
        )
        media_tensor = torch.nn.functional.pad(media_tensor, padding)

    elif torch.is_tensor(media_path):
        media_tensor = media_path.unsqueeze(0)
        num_input_frames = media_tensor.shape[2]
    elif isinstance(media_path, str) and any( media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
        reader = imageio.get_reader(media_path)
        num_input_frames = min(reader.count_frames(), max_frames)

        # Read and preprocess the relevant frames from the video file.
        frames = []
        for i in range(num_input_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame_tensor = load_image_to_tensor_with_resize_and_crop(
                frame, height, width, just_crop=just_crop
            )
            frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
            frames.append(frame_tensor)
        reader.close()

        # Stack frames along the temporal dimension
        media_tensor = torch.cat(frames, dim=2)
    else:
        raise Exception("video format not supported")
    return media_tensor


if __name__ == "__main__":
    main()