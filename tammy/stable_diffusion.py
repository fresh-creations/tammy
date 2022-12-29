import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline
import os
import subprocess

class StableDiffuser:
    def __init__(self, device, img_gen_settings) -> None:
        self.device = device
        size = img_gen_settings['size']
        self.width = size[0]
        self.heigth = size[1]


        if not os.path.exists('./checkpoints/stable-diffusion-v1-5'):
            os.chdir("./checkpoints")
            env = {
                **os.environ,
                "GIT_LFS_SKIP_SMUDGE": str(1),
            }
            subprocess.Popen(["git", "clone", "https://huggingface.co/runwayml/stable-diffusion-v1-5"], env=env).wait()
            os.chdir("./stable-diffusion-v1-5")
            subprocess.call('git lfs pull --include text_encoder/pytorch_model.bin', shell=True)
            os.chdir("./text_encoder")
            subprocess.call('ls -lh', shell=True)
            os.chdir("..")
            subprocess.call('git lfs pull --include vae/diffusion_pytorch_model.bin', shell=True)
            subprocess.call('git lfs pull --include unet/diffusion_pytorch_model.bin', shell=True)
            os.chdir("../..")

    def get_image(self, frame, img_0, step_dir,prompt, image_prompts, noise_prompt_seeds, 
                    noise_prompt_weights, iterations_per_frame, save_all_iterations):

        model_path = "./checkpoints/stable-diffusion-v1-5"
        print('prompt:', prompt)
        prompt = prompt[0]

        if frame == 0:

            #first image
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path, 
                #revision="fp16", 
                safety_checker=None,
                #torch_dtype=torch.float16,
            )
            pipe = pipe.to(self.device)

            pipe.enable_attention_slicing()
            image = pipe(prompt, height=self.heigth, width=self.width,num_inference_steps=10)
            image.images[0].save(os.path.join(step_dir,f"{frame+1:06}.png"))

        elif frame > 0:
            
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_path,
                #revision="fp16", 
                safety_checker=None,
                #torch_dtype=torch.float16,
            )
            # or download via git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
            # and pass `model_id_or_path="./stable-diffusion-v1-5"`.
            pipe = pipe.to(self.device)
            pipe.enable_attention_slicing()

            # let's download an initial image
            
            init_image = Image.fromarray(img_0)
            iterations_per_frame = 2
            images = pipe(prompt=prompt, image=init_image, num_inference_steps=iterations_per_frame, strength=0.1, guidance_scale=7.5)

            images.images[0].save(os.path.join(step_dir,f"{frame+1:06}.png"))