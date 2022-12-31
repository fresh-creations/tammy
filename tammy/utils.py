import os
import shutil
import cv2
import torch
import ffmpeg
import gc
from datetime import datetime
from PIL import ImageFile
import wget

ImageFile.LOAD_TRUNCATED_IMAGES = True

def init_exp(exp_name, text_prompts, working_dir, config_path):
    exp_base_dir = os.path.join(working_dir,'experiments')
    if not os.path.exists(exp_base_dir):
        os.mkdir(exp_base_dir)

    checkpoint_dir = os.path.join(working_dir,'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)


    if not os.path.exists(os.path.join(exp_base_dir,exp_name)):
        os.mkdir(os.path.join(exp_base_dir,exp_name))

    partial_key_frame = text_prompts[0:10].replace("'", "").replace(' ', '')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    foldername = f'{timestamp}-{partial_key_frame}'
    exp_dir = os.path.join(exp_base_dir,exp_name,foldername)
    os.mkdir(os.path.join(exp_base_dir,exp_name,foldername))
    os.mkdir(os.path.join(exp_base_dir,exp_name,foldername,'steps'))

    settings_file = os.path.join(exp_dir,'settings.yaml')
    shutil.copyfile(config_path, os.path.join(settings_file))
    step_dir = os.path.join(exp_dir,'steps')

    return exp_dir, foldername, settings_file, step_dir

def download_img(img_url):
    try:
        return wget.download(img_url,out="input.jpg")
    except:
        return

def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def empty_cuda():
    try:
        gc.collect()
    except:
        pass
    try:
        torch.cuda.empty_cache()
    except:
        pass

model_names={
    "vqgan_imagenet_f16_16384": 'ImageNet 16384',
    "vqgan_imagenet_f16_1024":"ImageNet 1024", 
    "wikiart_1024":"WikiArt 1024",
    "wikiart_16384":"WikiArt 16384",
    "coco":"COCO-Stuff",
    "faceshq":"FacesHQ",
    "sflckr":"S-FLCKR"
}

def export_to_ffmpeg(image_path, fps, output_path):
    (
        ffmpeg
        .input(image_path, pattern_type='glob', framerate=fps)
        .output(output_path,crf=17,preset='slower', pix_fmt='yuv420p',vcodec='libx264')
        .run()
    )
    print("Video after super-resolution is ready")

