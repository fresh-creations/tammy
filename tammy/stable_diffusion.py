import os
import subprocess
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def make_scheduler(num_inference_steps, from_scheduler=None):
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", steps_offset=1)
    scheduler.set_timesteps(num_inference_steps)
    if from_scheduler:
        scheduler.cur_model_output = from_scheduler.cur_model_output
        scheduler.counter = from_scheduler.counter
        scheduler.cur_sample = from_scheduler.cur_sample
        scheduler.ets = from_scheduler.ets[:]
    return scheduler


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    # when interpolating in latent spaces of something like a VAE or a GAN where you assume the distribution
    # was Gaussian, always interpolate in polar coordinates, rather than in cartesian coordinates
    # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.detach().cpu().numpy()
        v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


class StableDiffuser:
    def __init__(self, device, img_settings) -> None:
        self.device = device
        size = img_settings["size"]
        self.width = size[0]
        self.heigth = size[1]
        self.model_path = "./checkpoints/stable-diffusion-v1-5"

        if not os.path.exists(self.model_path):
            self.fetch_model()

    def pipe_init(self):
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            revision="fp16" if self.device == torch.device("cuda:0") else "fp32",
            safety_checker=None,
            torch_dtype=torch.float16 if self.device == torch.device("cuda:0") else torch.float32,
        )
        self.sd_pipe = self.sd_pipe.to(self.device)

        self.sd_pipe.enable_attention_slicing()

    def pipe_del(self):
        del self.sd_pipe

    def img2img_init(self):
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_path,
            revision="fp16" if self.device == torch.device("cuda:0") else "fp32",
            safety_checker=None,
            torch_dtype=torch.float16 if self.device == torch.device("cuda:0") else torch.float32,
        )
        self.img2img_pipe = self.img2img_pipe.to(self.device)
        self.img2img_pipe.enable_attention_slicing()

    def img2img_del(self):
        del self.img2img_pipe

    @staticmethod
    def fetch_model():
        os.chdir("./checkpoints")
        env = {
            **os.environ,
            "GIT_LFS_SKIP_SMUDGE": str(1),
        }
        subprocess.Popen(["git", "clone", "https://huggingface.co/runwayml/stable-diffusion-v1-5"], env=env).wait()
        os.chdir("./stable-diffusion-v1-5")
        subprocess.call("git lfs pull --include text_encoder/pytorch_model.bin", shell=True)
        os.chdir("./text_encoder")
        subprocess.call("ls -lh", shell=True)
        os.chdir("..")
        subprocess.call("git lfs pull --include vae/diffusion_pytorch_model.bin", shell=True)
        subprocess.call("git lfs pull --include unet/diffusion_pytorch_model.bin", shell=True)
        os.chdir("../..")

    def get_image(
        self,
        frame,
        img_0,
        step_dir,
        prompt,
        image_prompts,
        noise_prompt_seeds,
        noise_prompt_weights,
        iterations_per_frame,
        save_all_iterations,
    ):

        print("prompt:", prompt)
        prompt = prompt[0]

        if frame == 0:
            self.pipe_init()
            image = self.sd_pipe(prompt, height=self.heigth, width=self.width, num_inference_steps=10)
            image.images[0].save(os.path.join(step_dir, f"{frame+1:06}.png"))
            self.pipe_del()
            self.img2img_init()

        elif frame > 0:

            init_image = Image.fromarray(img_0)
            iterations_per_frame = 2
            images = self.img2img_pipe(
                prompt=prompt,
                init_image=init_image,
                num_inference_steps=iterations_per_frame,
                strength=0.1,
                guidance_scale=7.5,
            )

            images.images[0].save(os.path.join(step_dir, f"{frame+1:06}.png"))


class CustomStableDiffuser:
    def __init__(self, device, img_settings) -> None:
        model_path = "./checkpoints/stable-diffusion-v1-5"

        if not os.path.exists(model_path):
            StableDiffuser.fetch_model()

        self.device = device
        self.batch_size = 1
        size = img_settings["size"]
        self.width = size[0]
        self.heigth = size[1]
        self.pipe = StableDiffusionAnimationPipeline.from_pretrained(
            model_path,
            revision="fp16" if self.device == torch.device("cuda:0") else "fp32",
            safety_checker=None,
            torch_dtype=torch.float16 if self.device == torch.device("cuda:0") else torch.float32,
        )
        self.pipe.to(self.device)

        self.pipe.enable_attention_slicing()

    def interpolate_latents(self, frames_latents, num_interpolation_steps):
        print("Interpolating images from latents")
        images = []
        for i in range(len(frames_latents) - 1):
            latents_start = frames_latents[i]
            latents_end = frames_latents[i + 1]
            for j in range(num_interpolation_steps):
                x = j / num_interpolation_steps
                latents = latents_start * (1 - x) + latents_end * x
                if self.device == torch.device("cuda:0"):
                    latents = latents.half()
                image = self.pipe.latents_to_image(latents)
                images.append(image)
        return images

    def init_scheduler(self, num_inference_steps, prompt_strength, height, width, guidance_scale):
        seed = None
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        generator = torch.Generator(self.device).manual_seed(seed)

        # Generate initial latents to start to generate animation frames from
        initial_scheduler = self.pipe.scheduler = make_scheduler(num_inference_steps)

        num_initial_steps = int(num_inference_steps * (1 - prompt_strength))
        print(f"Generating initial latents for {num_initial_steps} steps")
        initial_latents = torch.randn(
            (self.batch_size, self.pipe.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
        )
        do_classifier_free_guidance = guidance_scale > 1.0

        self.pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)

        return initial_latents, do_classifier_free_guidance, num_initial_steps

    def init_latents(self, prompts, guidance_scale, num_inference_steps, prompt_strength):

        initial_latents, do_classifier_free_guidance, num_initial_steps = self.init_scheduler(
            num_inference_steps, prompt_strength, self.heigth, self.width, guidance_scale
        )

        keyframe_text_embeddings = []

        for prompt in prompts:
            keyframe_text_embeddings.append(self.pipe.embed_text(prompt, do_classifier_free_guidance, self.batch_size))

        if len(prompts) % 2 == 0:
            i = len(prompts) // 2 - 1
            prev_text_emb = keyframe_text_embeddings[i]
            next_text_emb = keyframe_text_embeddings[i + 1]
            text_embeddings_mid = slerp(0.5, prev_text_emb, next_text_emb)
        else:
            i = len(prompts) // 2
            text_embeddings_mid = keyframe_text_embeddings[i]

        latents_mid = self.pipe.denoise(
            latents=initial_latents,
            text_embeddings=text_embeddings_mid,
            t_start=1,
            t_end=num_initial_steps,
            guidance_scale=guidance_scale,
        )

        initial_scheduler = self.pipe.scheduler = make_scheduler(num_inference_steps)

        return latents_mid, keyframe_text_embeddings, num_initial_steps, initial_scheduler

    def get_image(
        self, latents_mid, text_embeddings, guidance_scale, num_inference_steps, initial_scheduler, num_initial_steps
    ):

        self.pipe.scheduler = make_scheduler(num_inference_steps, initial_scheduler)

        latents = self.pipe.denoise(
            latents=latents_mid,
            text_embeddings=text_embeddings,
            t_start=num_initial_steps,
            t_end=None,
            guidance_scale=guidance_scale,
        )

        if self.device == torch.device("cuda:0"):
            latents = latents.half()
        image = self.pipe.latents_to_image(latents)

        img = self.pipe.numpy_to_pil(image)[0]
        return img


class StableDiffusionAnimationPipeline(DiffusionPipeline):
    """
    From https://github.com/huggingface/diffusers/pull/241
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def denoise(self, latents, text_embeddings, t_start, t_end, guidance_scale):
        do_classifier_free_guidance = guidance_scale > 1.0

        for i, t in enumerate(self.scheduler.timesteps[t_start:t_end]):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents

    def embed_text(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool,
        batch_size: int,
    ) -> torch.FloatTensor:
        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def latents_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image.sample / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        return image
