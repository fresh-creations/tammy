import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from tqdm import tqdm

from tammy.stable_diffusion import (
    CustomStableDiffuser,
    StableDiffuser,
    make_scheduler,
    slerp,
)
from tammy.utils import read_image_workaround
from tammy.vqgan_clip import VQGAN_CLIP


def interpolation_scheduler(prompts, its_per_frame):
    """iteration per frame calculation
    we need to divide p prompt over f frames
    every frame has number of iterations its
    """
    total_its = sum(its_per_frame)
    logging.info(f"total_its: {total_its}")
    logging.info(f"nr_prompts: {len(prompts)}")

    num_animation_frames_series = []
    logging.info(f"its_per_frame: {its_per_frame}")
    it_budget_per_prompt = int(total_its / (len(prompts) - 1))
    assert max(its_per_frame) < it_budget_per_prompt, "prompt_transition larger then iteration budget per prompt"
    logging.info(f"it_budget_per_prompt: {it_budget_per_prompt}")
    prev_idx = 0
    for prompt_idx in range(len(prompts) - 1):
        cum_its = 0
        for idx, frame_its in enumerate(its_per_frame[prev_idx::]):
            print("idx", idx, "frame_its", frame_its, "cum_its", cum_its)
            if (cum_its + frame_its) >= it_budget_per_prompt:
                num_animation_frames_series.append(idx)
                prev_idx = idx
                break
            cum_its += frame_its

    return num_animation_frames_series


def warp(img_0, angle, zoom, translation_x, translation_y):
    """
    This function applies a combination of rotation, zoom, and translation to an image.
    It uses the OpenCV library to perform the transformation.

    Parameters:
        img_0 (numpy array): The original image to be transformed.
        angle (float): The angle of rotation to be applied to the image in degrees.
        zoom (float): The zoom factor to be applied to the image.
        translation_x (int): The number of pixels to translate the image in the x-axis.
        translation_y (int): The number of pixels to translate the image in the y-axis.

    Returns:
        transformed_img (numpy array): The transformed image.
    """

    center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)

    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    transformation_matrix = np.matmul(rot_mat, trans_mat)

    transformed_img = cv2.warpPerspective(
        img_0, transformation_matrix, (img_0.shape[1], img_0.shape[0]), borderMode=cv2.BORDER_WRAP
    )

    return transformed_img


class Animator2D:
    def __init__(
        self, model_type, img_gen_settings, device, max_frames, initial_image, step_dir, save_all_iterations
    ) -> None:
        self.device = device
        self.max_frames = max_frames
        self.key_frames = True
        self.save_all_iterations = save_all_iterations
        self.initial_image = initial_image
        self.step_dir = step_dir

        if model_type == "vqgan":
            self.generator = VQGAN_CLIP(img_gen_settings, device)
        elif model_type == "stable_diffusion":
            self.generator = StableDiffuser(device, img_gen_settings)

    def run(
        self,
        iterations_per_frame,
        angle_series,
        zoom_series,
        translation_x_series,
        translation_y_series,
        target_images_series,
        text_prompts_series,
        iterations_per_frame_series,
        noise_prompt_seeds,
        noise_prompt_weights,
    ):
        times = []
        its_to_do = sum(iterations_per_frame_series.values[0 : self.max_frames])
        total_its = 0
        start_time = datetime.now()

        with tqdm(total=its_to_do) as pbar:
            for i in range(self.max_frames):
                pbar.set_description(f"generating frames : {i}/{self.max_frames}")

                text_prompts = text_prompts_series[i]
                # convert single prompt string to list of strings
                text_prompts = [phrase.strip() for phrase in text_prompts.split("|")]
                if text_prompts == [""]:
                    text_prompts = []
                self.prompts = text_prompts

                target_images = target_images_series[i]

                if target_images == "None" or not target_images:
                    target_images = []
                else:
                    target_images = target_images.split("|")
                    target_images = [image.strip() for image in target_images]
                self.image_prompts = target_images

                angle = angle_series[i]
                zoom = zoom_series[i]
                zoom = 1.03
                translation_x = translation_x_series[i]
                translation_y = translation_y_series[i]
                iterations_per_frame = iterations_per_frame_series[i]

                if i > 0:
                    if self.save_all_iterations:
                        img_0 = read_image_workaround(f"{self.step_dir}/{i:06d}_{iterations_per_frame}.png")
                    else:
                        img_0 = read_image_workaround(f"{self.step_dir}/{i:06d}.png")

                    # warp loaded image
                    img_0 = warp(img_0, angle, zoom, translation_x, translation_y)

                else:
                    img_0 = None

                self.generator.get_image(
                    i,
                    img_0,
                    self.step_dir,
                    self.prompts,
                    self.image_prompts,
                    noise_prompt_seeds,
                    noise_prompt_weights,
                    iterations_per_frame,
                    self.save_all_iterations,
                )

                total_its += iterations_per_frame
                pbar.update(iterations_per_frame)
                time_elapsed = datetime.now() - start_time
                time_per_it = time_elapsed / total_its
                remaining_its = sum(iterations_per_frame_series.values[i::])
                remaining_time = remaining_its * time_per_it
                times.append(remaining_time)


class AnimatorInterpolate:
    def __init__(
        self, model_type, img_settings, device, max_frames, initial_image, step_dir, save_all_iterations
    ) -> None:
        self.device = device
        self.max_frames = max_frames
        self.key_frames = True
        self.save_all_iterations = save_all_iterations
        self.initial_image = initial_image
        self.step_dir = step_dir

        if model_type == "stable_diffusion":
            print("img_settings", img_settings)
            self.generator = CustomStableDiffuser(device, img_settings)
        else:
            print(f"AnimatorInterpolate not implemented for {model_type}")

    def run(self, text_prompts_series, iterations_per_frame_series, guidance_scale_series, prompt_strength_series):

        # dont use interations for last frame since we interpolate between different frames
        iterations_per_frame_values = iterations_per_frame_series.values[0:-1]
        prompts = text_prompts_series
        logging.info(f"prompts: {prompts}")
        prompt_strength = prompt_strength_series[0]
        guidance_scale = guidance_scale_series[0]
        num_inference_steps = iterations_per_frame_values[0]
        logging.info(f"num_inference_steps: {num_inference_steps}")
        print("iterations_per_frame_series", iterations_per_frame_values)
        initial_scheduler = self.generator.pipe.scheduler = make_scheduler(num_inference_steps)

        num_animation_frames_series = interpolation_scheduler(prompts, iterations_per_frame_values)

        with torch.no_grad():
            latents_mid, keyframe_text_embeddings, num_initial_steps, initial_scheduler = self.generator.init_latents(
                prompts, guidance_scale, num_inference_steps, prompt_strength
            )
            it_end_prev = 0
            # Generate animation frames
            frame_number = 1
            iteration = 0
            print("num_animation_frames_series", num_animation_frames_series)
            for keyframe in range(len(prompts) - 1):
                print("keyframe", keyframe)
                num_animation_frames = num_animation_frames_series[keyframe]
                cum_its = 0
                start_it = it_end_prev
                end_it = start_it + num_animation_frames
                it_end_prev = end_it
                its_per_frame = np.asarray(iterations_per_frame_values[start_it:end_it])
                total_its = np.sum(its_per_frame)
                print("prompt_strength_series", prompt_strength_series)
                print("num_animation_frames", num_animation_frames)
                for i in tqdm(range(num_animation_frames)):
                    logging.info(f"cum_its: {cum_its}")
                    logging.info(f"total_its: {total_its}")

                    iteration += 1
                    print("frame", frame_number)
                    iteration = num_animation_frames * keyframe
                    print("iteration", iteration)
                    prompt_strength = prompt_strength_series[0]
                    guidance_scale = guidance_scale_series[0]

                    logging.info(f"Generating frame {i} of keyframe {keyframe} with interp {cum_its/total_its}")
                    text_embeddings = slerp(
                        cum_its / total_its,
                        keyframe_text_embeddings[keyframe],
                        keyframe_text_embeddings[keyframe + 1],
                    )
                    if i < len(its_per_frame):
                        cum_its += its_per_frame[i]

                    img = self.generator.get_image(
                        latents_mid,
                        text_embeddings,
                        guidance_scale,
                        num_inference_steps,
                        initial_scheduler,
                        num_initial_steps,
                    )
                    if 1:
                        draw = ImageDraw.Draw(img)

                        text = (
                            f"{prompts[keyframe]} to \n {prompts[keyframe+1]} \n with {round((cum_its / total_its),4)}"
                        )
                        font = ImageFont.truetype("DejaVuSans.ttf", 25)
                        draw.text((0, 0), text, (255, 255, 255), font=font)
                    img.save(os.path.join(self.step_dir, f"{frame_number:06d}.png"))
                    frame_number += 1
                print("keyframe", keyframe, "prompt_len", (len(prompts) - 2))
                if keyframe == (len(prompts) - 2):
                    print("last", frame_number)
                    text_embeddings = slerp(
                        1,
                        keyframe_text_embeddings[keyframe],
                        keyframe_text_embeddings[keyframe + 1],
                    )

                    img = self.generator.get_image(
                        latents_mid,
                        text_embeddings,
                        guidance_scale,
                        num_inference_steps,
                        initial_scheduler,
                        num_initial_steps,
                    )

                    img.save(os.path.join(self.step_dir, f"{frame_number:06d}.png"))
                    frame_number += 1

        del self.generator
