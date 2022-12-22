from tammy.utils import read_image_workaround
from tammy.vqgan_clip import VQGAN_CLIP
from datetime import datetime
import cv2
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

class SequenceMaker:

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