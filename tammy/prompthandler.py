import ast
import re
import numpy as np
import pandas as pd

def get_inbetweens(key_frames_dict,max_frames, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.
    Parameters
    ----------
    key_frames_dict: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.
    
    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.
    
    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64
    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    for i, value in key_frames_dict.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)
    key_frame_series = key_frame_series.interpolate(limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.
    
    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.
    
    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """

    try:
        # This is the preferred way, the regex way will eventually be deprecated.
        frames = ast.literal_eval('{' + string + '}')
        if isinstance(frames, set):
            # If user forgot keyframes, just set value of frame 0
            (frame,) = list(frames)
            frames = {0: frame}
        return frames
    except Exception:
        import re
        pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
        frames = dict()
        for match_object in re.finditer(pattern, string):
            frame = int(match_object.groupdict()['frame'])
            param = match_object.groupdict()['param']
            if prompt_parser:
                frames[frame] = prompt_parser(param)
            else:
                frames[frame] = param

        if frames == {} and len(string) != 0:
            raise RuntimeError(f'Key Frame string not correctly formatted: {string}')
        return frames


def calc_its(zoom, its,min_zoom, max_zoom, its_min, its_max):
    slope = (its_max-its_min)/(max_zoom-min_zoom)
    intercept = its_max-slope*max_zoom
    if its<its_max:
        new_its = zoom*slope + intercept
    else:
        new_its = its
    return int(new_its)

class PromptHandler:
    def __init__(self) -> None:
        self.a = 1

    def handle(self, angle, translation_x, translation_y, iterations_per_frame, text_prompts, 
               target_images, max_frames, zoom_scale_factor, zoom_instrument, initial_fps,
               min_zoom=1, max_zoom=1.05, its_min=1, its_max=3):

        key_frames = True

        if len(zoom_instrument) > 0:
            with open(f'instruments/{zoom_instrument}_{initial_fps}.txt') as f:
                keyframe_beat = f.readlines()[0]
            match = re.search(f'{max_frames}:', keyframe_beat)
            letter_idx = match.start()
            zoom = keyframe_beat[0:letter_idx - 2]
        else:
            zoom = "1.05"

        parameter_dicts = dict()
        parameter_dicts['zoom'] = parse_key_frames(zoom, prompt_parser=float)
        parameter_dicts['angle'] = parse_key_frames(angle, prompt_parser=float)
        parameter_dicts['translation_x'] = parse_key_frames(translation_x, prompt_parser=float)
        parameter_dicts['translation_y'] = parse_key_frames(translation_y, prompt_parser=float)
        parameter_dicts['iterations_per_frame'] = parse_key_frames(iterations_per_frame, prompt_parser=int)

        text_prompts_dict = parse_key_frames(text_prompts)
        for key, value in list(text_prompts_dict.items()):
            parameter_dicts[f'text_prompt: {key}'] = value

        image_prompts_dict = parse_key_frames(target_images)
        for key, value in list(image_prompts_dict.items()):
            parameter_dicts[f'image_prompt: {key}'] = value

        if key_frames:
            text_prompts_series_dict = dict()
            for parameter in parameter_dicts.keys():
                if len(parameter_dicts[parameter]) > 0:
                    if parameter.startswith('text_prompt:'):
                        try:
                            text_prompts_series_dict[parameter] = get_inbetweens(parameter_dicts[parameter],max_frames)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "WARNING: You have selected to use key frames, but you have not "
                                "formatted `text_prompts` correctly for key frames.\n"
                                "Please read the instructions to find out how to use key frames "
                                "correctly.\n"
                            )
            text_prompts_series = pd.Series([np.nan for a in range(max_frames)])
            for i in range(max_frames):
                combined_prompt = []
                for parameter, value in text_prompts_series_dict.items():
                    parameter = parameter[len('text_prompt:'):].strip()
                    combined_prompt.append(f'{parameter}: {value[i]}')
                text_prompts_series[i] = ' | '.join(combined_prompt)

            image_prompts_series_dict = dict()
            for parameter in parameter_dicts.keys():
                if len(parameter_dicts[parameter]) > 0:
                    if parameter.startswith('image_prompt:'):
                        try:
                            image_prompts_series_dict[parameter] = get_inbetweens(parameter_dicts[parameter],max_frames)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "WARNING: You have selected to use key frames, but you have not "
                                "formatted `image_prompts` correctly for key frames.\n"
                                "Please read the instructions to find out how to use key frames "
                                "correctly.\n"
                            )
                            
            target_images_series = pd.Series([np.nan for a in range(max_frames)])
            for i in range(max_frames):
                combined_prompt = []
                for parameter, value in image_prompts_series_dict.items():
                    parameter = parameter[len('image_prompt:'):].strip()
                    combined_prompt.append(f'{parameter}: {value[i]}')
                target_images_series[i] = ' | '.join(combined_prompt)

            try:
                angle_series = get_inbetweens(parameter_dicts['angle'],max_frames)
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `angle` correctly for key frames.\n"
                    )

            try:
                zoom_series = get_inbetweens(parameter_dicts['zoom'],max_frames)
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `zoom` correctly for key frames.\n"
                    )

            for i, zoom in enumerate(zoom_series):
                if zoom <= 0:
                    print(
                        f"WARNING: You have selected a zoom of {zoom} at frame {i}. "
                        "This is meaningless. "
                        "If you want to zoom out, use a value between 0 and 1. "
                        "If you want no zoom, use a value of 1."
                    )

            try:
                translation_x_series = get_inbetweens(parameter_dicts['translation_x'],max_frames)
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_x` correctly for key frames.\n"
                    )

            try:
                translation_y_series = get_inbetweens(parameter_dicts['translation_y'],max_frames)
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_y` correctly for key frames.\n"
                    )

            try:
                iterations_per_frame_series = get_inbetweens(
                    parameter_dicts['iterations_per_frame'],max_frames, integer=True
                )
            except RuntimeError as e:
                print(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `iterations_per_frame` correctly for key frames.\n"
                    )

        else:
            text_prompts = [phrase.strip() for phrase in text_prompts.split("|")]
            if text_prompts == ['']:
                text_prompts = []
            if target_images == "None" or not target_images:
                target_images = []
            else:
                target_images = target_images.split("|")
                target_images = [image.strip() for image in target_images]

            iterations_per_frame = int(iterations_per_frame)


        zoom_series = (zoom_series - 1) * zoom_scale_factor + 1

        zooms = zoom_series.values
        iters = iterations_per_frame_series.values
        for zoom_idx, zoom in enumerate(zooms):
            its = iters[zoom_idx]
            iterations_per_frame_series.values[zoom_idx] =  calc_its(zoom=zoom,its=its,min_zoom=min_zoom, max_zoom=max_zoom, its_min=its_min, its_max=its_max)
        print(iterations_per_frame_series.values)
        sequence_settings = {
            'iterations_per_frame':iterations_per_frame,
            'angle_series':angle_series,
            'zoom_series':zoom_series,
            'translation_x_series':translation_x_series,
            'translation_y_series':translation_y_series,
            'text_prompts_series':text_prompts_series,
            'target_images_series':target_images_series,
            'iterations_per_frame_series': iterations_per_frame_series
        }
        # FIXME: currently below settings are not used
        sequence_settings['noise_prompt_seeds'] = []
        sequence_settings['noise_prompt_weights'] = []
        return sequence_settings
