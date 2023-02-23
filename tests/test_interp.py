import logging

import numpy as np
from tqdm import tqdm

from tammy.sequence_maker import interpolation_scheduler


def test_scheduler():
    prompts = ["horse", "dog", "men", "cat"]
    iterations_per_frame_values = [8, 10, 4, 30, 2, 15, 8, 8, 8]
    # nr_frames = len(iterations_per_frame_values)
    num_animation_frames_series = interpolation_scheduler(prompts, iterations_per_frame_values)

    print("num_animation_frames_series", num_animation_frames_series)
    it_end_prev = 0
    # Generate animation frames
    frame_number = 1
    iteration = 0

    for keyframe in range(len(prompts) - 1):
        num_animation_frames = num_animation_frames_series[keyframe]
        cum_its = 0
        start_it = it_end_prev
        end_it = start_it + num_animation_frames
        it_end_prev = end_it
        its_per_frame = np.asarray(iterations_per_frame_values[start_it:end_it])
        total_its = np.sum(its_per_frame)

        for i in tqdm(range(num_animation_frames)):
            logging.info(f"cum_its: {cum_its}")
            logging.info(f"total_its: {total_its}")

            iteration += 1
            print("frame", frame_number)

            ratio = cum_its / total_its
            logging.info(f"Generating frame {i} of keyframe {keyframe} with interp {ratio}")

            assert 1 > ratio >= 0

            if i < len(its_per_frame):
                cum_its += its_per_frame[i]

            frame_number += 1
