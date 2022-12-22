# Tammy
Tammy is an open-source python/pytorch project for music-based video generation with machine learning.
In other words, it allows to automatically generate videos taking into account various aspects of a song (e.g. BPM or piano pattern).


**Usage**  
For a quick start:
1. install tammy by cloning this repo and running `pip install .`
2. run `python run_tammy.py` which will use the default settings in `settings\settings_cpu.yaml`.

The `tammy` package can be easily used in your own script or other setting files can be used with the existing `run_tammy.py` script by running run `python run_tammy.py --settings_file <your_settings.yaml>`.

**Settings**  
The video generation has many configuration settings which can be specified in a `.yaml` file.

**Structure**
1. `tammy.prompthandler` generates the settings for every frame to be generated (e.g. translation or text prompt) based on a more concise description of the generation settings.  
2. `tammy.sequence_maker` has a `generator` which generates images based on a text prompt and input image (except for the the first image a fully new image is generated). Currently the only supported model is _VQGAN-CLIP_.
3. `tammy.upscaling` scales up the generated images with super-resolution. Currently the only supported model is _SwinIR_.  
4. `tammy.superslowmo` interpolates generated (optionally upscaled) images to increase the FPS without needing to generate every frame with a `sequence_maker`. Currently the only supported model is _SuperSloMo_.  

**Calculate video length**  

$$ \large  \frac{(frames-2) \cdot slowmo}{target \  fps}  $$
 
For example, with settings: frames=98, slowmo=5, target_fps=24, we will generate 20.0 seconds.

**Creating keyframes for instruments**  
Use `https://www.chigozie.co.uk/audio-keyframe-generator/` to generate keyframes based on audio. A good function for highlighting kick sound seems to be: `1 + 0.2 * x^4` but experimentation is required.

**Examples**  
The following video was generated using `tammy` (watch in 4K for best experience!).  
https://www.youtube.com/watch?v=T_bii9VLDk0

