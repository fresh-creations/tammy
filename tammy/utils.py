import gc
import os
import shutil
import tarfile
from datetime import datetime

import cv2
import ffmpeg
import gdown
import numpy as np
import torch
import wget
from mega import Mega
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
TAMMY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def init_exp(exp_name, text_prompts, working_dir, config_path):
    """
    This function initializes a new experiment by creating the necessary directories and
    copies the settings from a config file.

    Parameters:
        exp_name (str): The name of the experiment.
        text_prompts (str): The text prompts that will be used in the experiment.
        working_dir (str): The path of the working directory.
        config_path (str): The path of the config file.

    Returns:
        Tuple: containing
        exp_dir (str): The absolute path of the experiment directory.
        foldername (str): The relative path of the folder where the experiment will be stored.
        settings_file (str): The absolute path of the settings file.
        step_dir (str): The path of the directory where images are generated.
    """
    exp_base_dir = os.path.join(working_dir, "experiments")
    os.makedirs(exp_base_dir, exist_ok=True)

    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs(os.path.join(exp_base_dir, exp_name), exist_ok=True)

    partial_key_frame = text_prompts[0:10].replace("'", "").replace(" ", "")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    foldername = f"{timestamp}-{partial_key_frame}"
    exp_dir = os.path.join(exp_base_dir, exp_name, foldername)
    os.mkdir(os.path.join(exp_base_dir, exp_name, foldername))
    os.mkdir(os.path.join(exp_base_dir, exp_name, foldername, "steps"))

    settings_file = os.path.join(exp_dir, "settings.yaml")
    shutil.copyfile(config_path, os.path.join(settings_file))
    step_dir = os.path.join(exp_dir, "steps")

    return exp_dir, foldername, settings_file, step_dir


def download_img(img_url):
    try:
        return wget.download(img_url, out="input.jpg")
    except:  # noqa: E722
        return


def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)


def empty_cuda():
    """
    This function attempts to empty the GPU memory by calling the garbage collector and the
    torch.cuda.empty_cache() function.
    """
    try:
        gc.collect()
    except:  # noqa: E722
        pass
    try:
        torch.cuda.empty_cache()
    except:  # noqa: E722
        pass


model_names = {
    "vqgan_imagenet_f16_16384": "ImageNet 16384",
    "vqgan_imagenet_f16_1024": "ImageNet 1024",
    "wikiart_1024": "WikiArt 1024",
    "wikiart_16384": "WikiArt 16384",
    "coco": "COCO-Stuff",
    "faceshq": "FacesHQ",
    "sflckr": "S-FLCKR",
}


def export_to_ffmpeg(image_path, fps, output_path):
    """
    This function exports images to a video file using ffmpeg.
    It takes in a path of the images, fps and output video path.
    It uses ffmpeg library to combine the images and create a video file with the given fps and output path.

    Parameters:
    image_path (str): The path of the images to be exported to video.
    fps (int): The number of frames per second in the video.
    output_path (str): The path of the output video.

    """
    (
        ffmpeg.input(image_path, pattern_type="glob", framerate=fps)
        .output(output_path, crf=17, preset="slower", pix_fmt="yuv420p", vcodec="libx264")
        .run()
    )


class SourceSeparator:
    def __init__(self) -> None:
        spleet_dir = os.path.join(TAMMY_DIR, "pretrained_models")
        os.makedirs(spleet_dir, exist_ok=True)
        if not os.path.exists(os.path.join(spleet_dir, "5stems")):
            wget.download("https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz", out=spleet_dir)
            tar_file = os.path.join(spleet_dir, "5stems.tar.gz")
            # open file
            file = tarfile.open(tar_file)
            # extracting file
            file.extractall(os.path.join(spleet_dir, "5stems"))
            file.close()

        # self.separator = Separator("spleeter:5stems")
        # self.audio_loader = AudioAdapter.default()
        self.sample_rate = 44100
        # self.spleet_dir = os.path.join(TAMMY_DIR, "spleeted_instruments")
        # os.makedirs(self.spleet_dir, exist_ok=True)

    def separate(self, initial_fps, audio_clip_path, instrument):
        """
        Separates the audio clip at the provided path into its individual instruments
        and writes the amplitude of the specified instrument to a text file.

        Parameters:
            - initial_fps: Initial frames per second of the audio clip.
            - audio_clip_path: Path to the audio clip to be separated.
            - instrument: Name of the instrument to extract the amplitude of.

        Returns:
            filename
        """
        waveform, _ = self.audio_loader.load(audio_clip_path, sample_rate=self.sample_rate)
        prediction = self.separator.separate(waveform)
        slice_idx = int(self.sample_rate / initial_fps)
        stem = prediction[instrument][::slice_idx]
        result = np.abs(stem[:, 0]) + np.abs(stem[:, 1])
        result_list = result.tolist()
        string_to_write = ""
        for frame_idx, magn in enumerate(result_list):
            string_to_write += f"{frame_idx}: ({1+magn:.3f}), "

        filename = os.path.join(self.spleet_dir, f"{instrument}_{initial_fps}.txt")
        with open(filename, "w") as text_file:
            text_file.write(string_to_write)

        return filename


def download_from_mega(url, path, filename):
    mega = Mega()
    m = mega.login()

    ckpt_path = os.path.join(path, filename)
    if os.path.getsize(ckpt_path) < 10e6:
        print(f"downloading {filename}")
        m.download_url(url, path, filename)


def download_from_google_drive(url, path, filename):
    ckpt_path = os.path.join(path, filename)
    if os.path.getsize(ckpt_path) < 10e6:
        print(f"downloading {filename}")
        gdown.download(url, ckpt_path, quiet=False)
