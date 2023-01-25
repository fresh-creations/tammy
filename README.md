# Tammy
Tammy is an open-source project that harnesses the power of deep learning to generate visually stunning music videos. Written in Python and PyTorch, it allows users to automatically generate videos that are synchronized with various aspects of a song, such as its BPM or piano pattern. The project utilizes deep learning models at various stages of the video generation process, including frame generation with GANs, spatial upscaling with super-resolution models, and temporal upsampling with frame interpolation models. The aim of this project is to provide an easy-to-use pipeline for combining different models to create unique and captivating music videos.


**Quick start**
For a quick start:
1. make sure to have ffmpeg `sudo apt-get install ffmpeg` and git-lfs installed.
2. install tammy by cloning this repo and running `pip install .`
3. run `python run_tammy.py` which will use the default settings in `settings\settings_cpu.yaml`.

The `tammy` package can be easily used in your own script or other setting files can be used with the existing `run_tammy.py` script by running run `python run_tammy.py --settings_file <your_settings.yaml>`.

**Contributing**
1. Follow the installation guidelines in quick start
2. Install the required pre-commit hooks: `pre-commit install`
3. Add tests for all added features
4. Make a pull request from a new branch

**Animation modes**
1. Animation_2d
2. Interpolation

**Structure**
1. `tammy.prompthandler` generates the settings for every frame to be generated (e.g. translation or text prompt) based on a more concise description of the generation settings.
2. `tammy.sequence_maker` has a `generator` which generates an image sequence based on a text prompts. Currently the supported model are _VQGAN-CLIP_ and _Stable-Diffusion_
3. `tammy.upscaling` scales up the generated images with super-resolution. Currently the only supported model is _SwinIR_.
4. `tammy.superslowmo` interpolates generated (optionally upscaled) images to increase the FPS without needing to generate every frame with a `sequence_maker`. Currently the only supported model is _SuperSloMo_.

**Settings**
The video generation has many configuration settings which can be specified in a `.yaml` file.

**Calculate video length**

$$ \large  \frac{(frames-2) \cdot slowmo}{target \  fps}  $$

For example, with settings: frames=98, slowmo=5, target_fps=24, we will generate 20.0 seconds.

**Creating keyframes for instruments**
Use `https://www.chigozie.co.uk/audio-keyframe-generator/` to generate keyframes based on audio. A good function for highlighting kick sound seems to be: `1 + 0.2 * x^4` but experimentation is required.

**Examples**
The following video was generated using `tammy` (watch in 4K for best experience!).
https://www.youtube.com/watch?v=T_bii9VLDk0
