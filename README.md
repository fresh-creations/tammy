# Tammy

Tammy is an open-source project focusing on music-based video generation with machine learning.
In other words, it allows to automatically generate videos taking into account various aspects of a song (e.g. kick).


**Usage**  

To `python run_tammy` which will use the default settings in `settings\settings_cpu.yaml`.

**Settings**

The video generation has many configuration settings which can be specified in a `.yaml` file.

**Calculate video length**  

$$ \large  \frac{(frames-2) \cdot slowmo}{target \  fps}  $$
 
For example, with settings: frames=98, slowmo=5, target_fps=24, we will generate 20.0 seconds.

**Creating keyframes**

Use `https://www.chigozie.co.uk/audio-keyframe-generator/` to generate keyframes based on audio. A good function for highlighting kick sound seems to be: `1 + 0.2 * x^4` but experimentation is required.

**Examples**
The following video was generated using `tammy`.
https://www.youtube.com/watch?v=T_bii9VLDk0&ab_channel=SkipsceneMusic