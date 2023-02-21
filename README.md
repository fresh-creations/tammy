# Tammy
Tammy is a Python/Pytorch-based open-source project that uses deep learning models to generate original music videos. It allows users to **automatically generate videos based on text prompt transitions that are synchronized with various aspects of a song**, such as its BPM or piano pattern. The project utilizes deep learning models at various stages of the video generation process, including audio source separation with LSTMs, frame generation with GANs, spatial upscaling with super-resolution models, and temporal upsampling with frame interpolation models. The aim of this project is to provide an easy-to-use framework to build custom model pipelines to create unique music videos.

### Example (turn on audio!) 

https://user-images.githubusercontent.com/28825134/219867875-d9ef07fa-1a27-49c5-9507-2c8b53257555.mp4


#### Table of Contents   
[Features](#features)   
[Quick start](#quick-start)   
[Dataflow and Code Structure](#dataflow-and-code-structure)   
[Generation Settings](#generation-settings)   
[More Examples](#more-examples)    
[Contributing](#contributing)    


## Features
- fully automated music-video generation. Provide just a song, text prompts and generation settings to generate a music-video.
- multiple animation modes: Animation_2d and Interpolation.
- automatic and lazy model loading. For example, the Stable Diffusion model is automatically fetched when used so no need to manually install any libraries or wait for all models to be downloaded when installing the package.


## Quick start  
For a quick start:
1. Make sure to install ffmpeg and libsndfile `sudo apt-get install ffmpeg libsndfile1` and [git-lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
2. Clone this repo
5. Install tammy by running `pip install .`
4. run `python run_tammy.py` which will use the default settings in `settings\settings_cpu.yaml` and default song `thoughtsarebeings_clip.wav`.

The `tammy` package can be easily used in your own script or other setting files and audio files can be used with the existing `run_tammy.py` script by running `python run_tammy.py --settings_file <your_settings.yaml> --audio_path <your_audio_file.mp3/wav>`.

## Dataflow and Code Structure
<img src="https://user-images.githubusercontent.com/28825134/219864907-f8e5608f-e50d-4fe8-ab4a-53babec48e72.svg" width="1500">  

1. `tammy.prompthandler` generates the settings for every frame to be generated (e.g. translation or text prompt) based on a more concise description of the generation settings.
2. `tammy.sequence_maker` has a `generator` which generates an image sequence based on a text prompts. Currently the supported models are _VQGAN-CLIP_ and _Stable-Diffusion_
3. `tammy.upscaling` scales up the generated images with super-resolution. Currently the only supported model is _SwinIR_.
4. `tammy.superslowmo` interpolates generated (optionally upscaled) images to increase the FPS without needing to generate every frame with a `sequence_maker`. Currently the only supported model is _SuperSloMo_.

## Generation Settings
The video generation has many configuration settings which are specified in a `<settings_file_name>.yaml` file. Some example setting files, mostly used for testing, can be found in the `settings` folder. Most setting names (keys in the `settings.yaml`) should be self-explanatory. For clarity, some settings are explained below.

### Instrument
Instruments are used to steer frame transitions, in particular: zoom in Animation_2d mode and prompt transition speed in Interpolation mode. `tammy` has two options to provide instruments: 
1. automatically by using Spleeter source separation: set `do_spleet: True` and provide `instrument: <instrument name>`
2. manually by providing a keyframe file in the setting file as `zoom_instrument: <file_name>` and name the file: `file_name_fps.txt` where `fps` should correspond with the `fps` value in `sequence_settings.initial_fps` . Keyframes can be manually generated with e.g. `https://www.chigozie.co.uk/audio-keyframe-generator/`

### Frame rate and video length
The setting `sequence_settings.initial_fps` determines the number of frames generated, given the length of the audio clip. By using frame interpolation, the frame-rate can be increased to a target by setting `do_slowmo: True` and providing a `target_fps` which must be a multiple of `initial_fps`. This allows to produce high frame rate videos faster than compared to generating all frames from scratch with the `generator`.

If desired, the number of generated frames can be limited by providing `sequence_settings.max_frames`. In this case the generated video length will be shorter than the provided audio clip and will be: `max_frames`/`initial_fps`. 

## More examples  
Video generated using VQGAN-CLIP and Animation_2d mode from `tammy`.

https://user-images.githubusercontent.com/28825134/219866498-1e770e8a-cfd9-412f-9657-433c0b499181.mp4

Full video (watch in 4K for best experience!: https://www.youtube.com/watch?v=T_bii9VLDk0  

Video generated using Stable Diffusion and Interpolation mode from `tammy`.


https://user-images.githubusercontent.com/28825134/219868015-7244512d-34f1-48d2-a9e7-690a96ef277d.mp4


## Contributing
1. Follow the installation guidelines in quick start.
2. Make an issue with your proposed feature.
3. Add your feature and accompanying tests in a new branch called `<your_name>\<feature>`.
4. Make sure all tests pass by running `pytest`.
5. Install the required pre-commit hooks: `pre-commit install`.
6. Make a pull request to merge into main and ask for a review.
