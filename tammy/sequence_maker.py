from tammy.utils import read_image_workaround
from tammy.vqgan_clip import VQGAN_CLIP
from tammy.stable_diffusion import StableDiffuser, CustomStableDiffuser, slerp, make_scheduler
from datetime import datetime
import cv2
import torch
import numpy as np
from tqdm import tqdm

def warp(img_0,angle,zoom,translation_x,translation_y):

    center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
    trans_mat = np.float32(
        [[1, 0, translation_x],
        [0, 1, translation_y]]
    )
    rot_mat = cv2.getRotationMatrix2D( center, angle, zoom )

    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    transformation_matrix = np.matmul(rot_mat, trans_mat)

    img_0 = cv2.warpPerspective(
        img_0,
        transformation_matrix,
        (img_0.shape[1], img_0.shape[0]),
        borderMode=cv2.BORDER_WRAP
    )

    return img_0

class Animator2D:

    def __init__(self,model_type,img_gen_settings, device, max_frames, initial_image, step_dir,
    save_all_iterations) -> None:
        self.device = device
        self.max_frames = max_frames
        self.key_frames = True
        self.save_all_iterations = save_all_iterations
        self.initial_image = initial_image
        self.step_dir = step_dir



        if model_type == 'vqgan':
            self.generator = VQGAN_CLIP(img_gen_settings, device)
        elif model_type == 'stable_diffusion':
            self.generator = StableDiffuser(device, img_gen_settings)

    def run(self, iterations_per_frame, angle_series, zoom_series, translation_x_series, translation_y_series, target_images_series,text_prompts_series,iterations_per_frame_series,
    noise_prompt_seeds, noise_prompt_weights):
           
        its_to_do = sum(iterations_per_frame_series.values[0:self.max_frames])
        total_its = 0
        times = []
        start_time = datetime.now()


        with tqdm(total=its_to_do) as pbar:
            for i in range(self.max_frames):
                pbar.set_description(f'generating frames : {i}/{self.max_frames}')

                text_prompts = text_prompts_series[i]
                # convert single prompt string to list of strings
                text_prompts = [phrase.strip() for phrase in text_prompts.split("|")]
                if text_prompts == ['']:
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
                        img_0 = read_image_workaround(
                            f'{self.step_dir}/{i:06d}_{iterations_per_frame}.png')
                    else:
                        img_0 = read_image_workaround(f'{self.step_dir}/{i:06d}.png')

                    # warp loaded image
                    img_0 = warp(img_0, angle, zoom, translation_x, translation_y)
                
                else:
                    img_0 = None


                self.generator.get_image(i, img_0, self.step_dir,self.prompts, self.image_prompts, noise_prompt_seeds, 
                                            noise_prompt_weights, iterations_per_frame, self.save_all_iterations)
                   
                total_its += iterations_per_frame   
                pbar.update(iterations_per_frame)                      
                time_elapsed = datetime.now()-start_time
                time_per_it = time_elapsed/total_its
                remaining_its = sum(iterations_per_frame_series.values[i::])
                remaining_time = remaining_its*time_per_it


class AnimatorInterpolate:
    
    def __init__(self,model_type,img_gen_settings, device, max_frames, initial_image, step_dir,
    save_all_iterations) -> None:
        self.device = device
        self.max_frames = max_frames
        self.key_frames = True
        self.save_all_iterations = save_all_iterations
        self.initial_image = initial_image
        self.step_dir = step_dir

        if model_type == 'stable_diffusion':
            self.generator = CustomStableDiffuser(device, img_gen_settings)
        else:
            print(f'AnimatorInterpolate not implemented for {model_type}')

    def run(self, zoom_series, iterations_per_frame,text_prompts_series,iterations_per_frame_series):


        prompt_start = 'tall rectangular black monolith, monkeys in the desert looking at a large tall monolith, a detailed matte painting by Wes Anderson, behance, light and space, reimagined by industrial light and magic, matte painting, criterion collection'
        prompt_end = """tall rectangular black monolith, monkeys astronaut in the desert looking at a large tall monolith, a detailed matte painting by Wes Anderson, behance, light and space, reimagined by industrial light and magic, matte painting, criterion collection
                    | a tall black spaceship lifting off the desert, a detailed matte painting by Wes Anderson, behance, light and space, reimagined by industrial light and magic, matte painting, criterion collection"""

        batch_size = 1
        num_inference_steps = 50
        initial_scheduler = self.generator.pipe.scheduler = make_scheduler(
            num_inference_steps
        )
        guidance_scale = 7.5
        num_animation_frames = 3
        prompt_strength = 0.85
        height = 512
        width = 512

        with torch.no_grad():
            print("Generating first and last keyframes")
            # re-initialize scheduler
            
            initial_latents, do_classifier_free_guidance, num_initial_steps = self.generator.init_scheduler(num_inference_steps, prompt_strength, height, width, guidance_scale)

            prompts = [prompt_start] + [
                p.strip() for p in prompt_end.strip().split("|")
            ]
            keyframe_text_embeddings = []
            keyframe_latents = []

            for prompt in prompts:
                keyframe_text_embeddings.append(
                    self.generator.pipe.embed_text(
                        prompt, do_classifier_free_guidance, batch_size
                    )
                )

            if len(prompts) % 2 == 0:
                i = len(prompts) // 2 - 1
                prev_text_emb = keyframe_text_embeddings[i]
                next_text_emb = keyframe_text_embeddings[i + 1]
                text_embeddings_mid = slerp(0.5, prev_text_emb, next_text_emb)
            else:
                i = len(prompts) // 2
                text_embeddings_mid = keyframe_text_embeddings[i]


            latents_mid = self.generator.pipe.denoise(
                latents=initial_latents,
                text_embeddings=text_embeddings_mid,
                t_start=1,
                t_end=num_initial_steps,
                guidance_scale=guidance_scale,
            )

            for prompt in [prompts[0], prompts[-1]]:
                keyframe_latents.append(
                    self.generator.pipe.denoise(
                        latents=latents_mid,
                        text_embeddings=keyframe_text_embeddings[i],
                        t_start=num_initial_steps,
                        t_end=None,
                        guidance_scale=guidance_scale,
                    )
                )

            # Generate animation frames
            frames_latents = []
            for keyframe in range(len(prompts) - 1):
                cum_its = 0
                start_it = keyframe*num_animation_frames
                end_it = (keyframe+1)*num_animation_frames
                its_per_frame = np.asarray(iterations_per_frame_series.values[start_it:end_it])
                total_its = np.sum(its_per_frame)
                for i in range(num_animation_frames):
                    print(f"Generating frame {i} of keyframe {keyframe} with interp {cum_its/total_its}")
                    text_embeddings = slerp(
                        cum_its / total_its,
                        keyframe_text_embeddings[keyframe],
                        keyframe_text_embeddings[keyframe + 1],
                    )
                    cum_its += its_per_frame[i] 

                    # re-initialize scheduler
                    self.generator.pipe.scheduler = make_scheduler(
                        num_inference_steps, initial_scheduler
                    )

                    latents = self.generator.pipe.denoise(
                        latents=latents_mid,
                        text_embeddings=text_embeddings,
                        t_start=num_initial_steps,
                        t_end=None,
                        guidance_scale=guidance_scale,
                    )

                    # de-noise this frame
                    frames_latents.append(latents.half())

                    image = self.generator.pipe.latents_to_image(latents.half())
                    import os

                    img = self.generator.pipe.numpy_to_pil(image)[0]
                    #path = f"/home/jonas/code/cog-stable-diffusion/output/steps/{(num_animation_frames*keyframe)+i:06d}.png"
                    img.save(os.path.join(self.step_dir,f"{(num_animation_frames*keyframe)+i:06d}.png"))

