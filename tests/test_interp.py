import logging
import os

import numpy as np
import yaml
from tqdm import tqdm

from tammy.prompthandler import PromptHandler
from tammy.sequence_maker import interpolation_scheduler

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")
TAMMY_DIR = os.path.dirname(TEST_DIR)


def test_scheduler():
    config_path = os.path.join(TAMMY_DIR, "settings/stable_diffusion_interpolate_test_cpu.yaml")
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    sequence_settings = config["sequence_settings"]
    animatation_mode = sequence_settings.pop("mode")
    prompt_handler = PromptHandler(animatation_mode)
    processed_sequence_settings = prompt_handler.handle(**sequence_settings)

    iterations_per_frame_series = processed_sequence_settings["iterations_per_frame_series"]
    iterations_per_frame_values = iterations_per_frame_series.values[0:-1]
    # prompts = ["horse", "dog", "men", "cat"]
    # iterations_per_frame_values = 1*[8, 10, 4, 30, 2, 15, 8, 8, 8]
    nr_frames = len(iterations_per_frame_values)

    prompts = processed_sequence_settings["text_prompts_series"]
    num_animation_frames_series = interpolation_scheduler(prompts, iterations_per_frame_values)
    nr_frames = len(num_animation_frames_series)
    it_end_prev = 0
    # Generate animation frames
    frame_number = 1
    iteration = 0

    for keyframe in range(len(prompts) - 1):
        num_animation_frames = num_animation_frames_series[keyframe]
        cum_its = 0
        start_it = it_end_prev
        end_it = max(start_it + num_animation_frames, nr_frames)
        logging.info(f"end_it: {end_it}")
        it_end_prev = end_it
        its_per_frame = np.asarray(iterations_per_frame_values[start_it:end_it])
        assert len(its_per_frame) > 0
        assert num_animation_frames > 0
        total_its = np.sum(its_per_frame)
        logging.info(f"num_animation_frames: {num_animation_frames}")
        logging.info(f"its_per_frame: {its_per_frame}")

        for i in tqdm(range(num_animation_frames)):
            logging.info(f"cum_its: {cum_its}")
            logging.info(f"total_its: {total_its}")

            iteration += 1

            ratio = cum_its / total_its
            logging.info(f"Generating frame {i} of keyframe {keyframe} with interp {ratio}")

            assert 1 > ratio >= 0

            if i < len(its_per_frame):
                assert its_per_frame[i] > 0
                cum_its += its_per_frame[i]

            frame_number += 1
    frame_number += 1
    assert frame_number == nr_frames
