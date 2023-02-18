# Tammy
Tammy is a Python/Pytorch-based open-source project that uses deep learning models to generate original music videos. It allows users to **automatically generate videos based on text prompt transitions that are synchronized with various aspects of a song**, such as its BPM or piano pattern. The project utilizes deep learning models at various stages of the video generation process, including audio source separation with LSTMs, frame generation with GANs, spatial upscaling with super-resolution models, and temporal upsampling with frame interpolation models. The aim of this project is to provide an easy-to-use framework to build custom model pipelines to create unique music videos.

### Example (turn on audio!) 
https://user-images.githubusercontent.com/28825134/219865723-87340d5f-3a21-4382-a0eb-48db82dfc92f.mp4  
  

<img src="https://user-images.githubusercontent.com/28825134/219864907-f8e5608f-e50d-4fe8-ab4a-53babec48e72.svg" width="1500">

## Features
- fully automated music-video generation. Provide just a song, text prompts and generation settings to generate a music-video.
- multiple animation modes: Animation_2d and Interpolation.
- automatic and lazy model loading. For example, the Stable Diffusion model is automatically fetched when used so no need to manually install any libraries or wait for all models to be downloaded when installing the package.


## Quick start  
For a quick start:
1. make sure to install ffmpeg and libsndfile `sudo apt-get install ffmpeg libsndfile1` and [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
2. install tammy by cloning this repo and running `pip install .`
3. run `python run_tammy.py` which will use the default settings in `settings\settings_cpu.yaml` and default song `thoughtsarebeings_clip.wav`.

The `tammy` package can be easily used in your own script or other setting files and audio files can be used with the existing `run_tammy.py` script by running `python run_tammy.py --settings_file <your_settings.yaml> --audio_path <your_audio_file.mp3/wav>`.

## Code Structure
1. `tammy.prompthandler` generates the settings for every frame to be generated (e.g. translation or text prompt) based on a more concise description of the generation settings.
2. `tammy.sequence_maker` has a `generator` which generates an image sequence based on a text prompts. Currently the supported models are _VQGAN-CLIP_ and _Stable-Diffusion_
3. `tammy.upscaling` scales up the generated images with super-resolution. Currently the only supported model is _SwinIR_.
4. `tammy.superslowmo` interpolates generated (optionally upscaled) images to increase the FPS without needing to generate every frame with a `sequence_maker`. Currently the only supported model is _SuperSloMo_.

## Generation Settings
The video generation has many configuration settings which can be specified in a `.yaml` file.

**Calculate video length**

$$ \large  \frac{(frames-2) \cdot slowmo}{target \  fps}  $$

For example, with settings: frames=98, slowmo=5, target_fps=24, we will generate 20.0 seconds.

**Creating keyframes for instruments**  
Use `https://www.chigozie.co.uk/audio-keyframe-generator/` to generate keyframes based on audio. A good function for highlighting kick sound seems to be: `1 + 0.2 * x^4` but experimentation is required.

## More examples  
The following video was generated using `tammy` (watch in 4K for best experience!).
https://www.youtube.com/watch?v=T_bii9VLDk0

## Contributing
1. Follow the installation guidelines in quick start.
2. Add your feature and accompanying tests.
3. Make sure all tests pass by running `pytest`.
3. Install the required pre-commit hooks: `pre-commit install`.
4. Make a pull request from a new branch and ask for a review.
