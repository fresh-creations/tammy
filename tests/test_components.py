import os
import subprocess
from shutil import rmtree

import cv2
import numpy as np
import pytest
from PIL import Image

from tammy.prompthandler import PromptHandler
from tammy.superslowmo.video_to_slomo import MotionSlower
from tammy.upscaling.super_resolution import Upscaler
from tammy.utils import SourceSeparator

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")
TAMMY_DIR = os.path.dirname(TEST_DIR)


def gen_test_imgs(width, height, number, dir):

    if not os.path.exists(dir):
        os.makedirs(dir)

    for img in range(number):
        image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

        im = Image.fromarray(image)
        im.save(os.path.join(dir, f"{img+1:06d}.png"))


def test_img_gen():

    gen_test_imgs(width=80, height=40, number=5, dir=TEST_DATA_DIR)
    nr_files = len(os.listdir(TEST_DATA_DIR))
    assert nr_files == 5
    rmtree(TEST_DATA_DIR)


@pytest.mark.parametrize("width, height, nr_input_frames, target_fps, slowmo_factor", [(256, 128, 5, 12, 4)])
def test_slowmo(width, height, nr_input_frames, target_fps, slowmo_factor):
    gen_test_imgs(width=width, height=height, number=nr_input_frames, dir=TEST_DATA_DIR)
    slowmo_settings = {"target_fps": target_fps, "slowmo_factor": slowmo_factor}
    motion_slower = MotionSlower(slowmo_settings=slowmo_settings, device="cpu", batch_size=1)
    video_name = os.path.join(TEST_DATA_DIR, "slowmo_vid.mp4")
    motion_slower.slomo(input_path=TEST_DATA_DIR, video_path=video_name)
    assert os.path.exists(video_name)

    video = cv2.VideoCapture(video_name)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_frames = (nr_input_frames - 1) * slowmo_factor
    assert expected_frames == total_frames
    os.remove(video_name)
    rmtree(TEST_DATA_DIR)


@pytest.mark.parametrize("width, height, nr_input_frames, upscale_factor, batch_size", [(80, 40, 5, 4, 1)])
def test_upscale(width, height, nr_input_frames, upscale_factor, batch_size):
    gen_test_imgs(width=width, height=height, number=nr_input_frames, dir=TEST_DATA_DIR)
    super_res_settings = {"upscale_factor": upscale_factor, "batch_size": batch_size}
    upscaler = Upscaler(super_res_settings, device="cpu")
    upscaler.upscale_w_swinir(TEST_DATA_DIR)
    super_res_dir = os.path.join(TEST_DATA_DIR, "super_res")
    nr_files = len(os.listdir(super_res_dir))
    assert nr_files == nr_input_frames

    first_frame = os.listdir(super_res_dir)[0]
    image = cv2.imread(os.path.join(super_res_dir, first_frame))
    upscaled_height, upscaled_width, channels = image.shape

    assert upscaled_height == upscale_factor * height
    assert upscaled_width == upscale_factor * width

    rmtree(super_res_dir)
    rmtree(TEST_DATA_DIR)


@pytest.mark.parametrize(
    "mode, initial_fps, max_frames, iterations_per_frame, text_prompts, prompt_strength, guidance_scale, zoom_instrument",
    [("interpolation", 6, 6, "0: 10", "dog,cat,horse", "0: 0.9", "0: 7.5", "kick")],
)
def test_prompt_handler(
    mode, initial_fps, max_frames, iterations_per_frame, text_prompts, prompt_strength, guidance_scale, zoom_instrument
):

    sequence_settings = {
        "mode": mode,
        "initial_fps": initial_fps,
        "max_frames": max_frames,
        "iterations_per_frame": iterations_per_frame,
        "text_prompts": text_prompts,
        "prompt_strength": prompt_strength,
        "guidance_scale": guidance_scale,
        "zoom_instrument": zoom_instrument,
    }

    animatation_mode = sequence_settings.pop("mode")
    prompt_handler = PromptHandler(animatation_mode)
    processed_sequence_settings = prompt_handler.handle(**sequence_settings)

    assert (processed_sequence_settings["iterations_per_frame_series"].values == np.array([20, 4, 4, 4, 4, 4])).any()


@pytest.mark.parametrize(
    "initial_fps, audio_clip_path, instrument", [(6, os.path.join(TAMMY_DIR, "thoughtsarebeings_clip.wav"), "drums")]
)
def test_source_sep(initial_fps, audio_clip_path, instrument):
    separator = SourceSeparator()
    filename = separator.separate(initial_fps, audio_clip_path, instrument)
    assert filename is not None


@pytest.mark.parametrize(
    "settings_file",
    [
        "stable_diffusion_interpolate_test_cpu.yaml",
        "stable_diffusion_animate_2d_test_cpu.yaml",
        "vqgan_clip_animate_2d_test_cpu.yaml",
    ],
)
def test_integration(settings_file):
    settings = os.path.join(TAMMY_DIR, "settings", settings_file)
    completed_process = subprocess.run(["python", os.path.join(TAMMY_DIR, "run_tammy.py"), "--settings_file", settings])
    assert completed_process.returncode == 0
