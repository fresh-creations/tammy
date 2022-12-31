import os
import torch
import yaml
import argparse
import warnings
from moviepy.editor import VideoFileClip, AudioFileClip
from tammy.sequence_maker import Animator2D, AnimatorInterpolate
from tammy.prompthandler import PromptHandler
from tammy.upscaling.super_resolution import Upscaler
from tammy.superslowmo.video_to_slomo import MotionSlower
from tammy.utils import init_exp, export_to_ffmpeg

#do some administration
warnings.filterwarnings("ignore")

#load all settings
parser = argparse.ArgumentParser(description='Run vqgan with settings')
parser.add_argument('--settings_file',default='settings/stable_diffusion_animate_2d_test_cpu.yaml.yaml')
args = parser.parse_args()
settings_path = args.settings_file

print('using settings', settings_path)

working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir,settings_path)

with open(config_path, 'r') as config_file:
    config =yaml.safe_load(config_file)

exp_settings = config['exp_settings']
sequence_settings = config['sequence_settings']
super_res_settings = config['super_res_settings']
slowmo_settings = config['slowmo_settings']
do_super_res = super_res_settings['do_super_res']
do_slowmo = slowmo_settings['do_slowmo']

exp_name = exp_settings['exp_name']
text_prompts = sequence_settings['text_prompts']
exp_dir, foldername, settings_file, step_dir = init_exp(exp_name, text_prompts, working_dir, config_path)

seed = exp_settings['seed']
save_all_iterations = exp_settings['save_all_iterations']

device = torch.device('cuda:0' if (torch.cuda.is_available() and (exp_settings['device']!='cpu')) else 'cpu')
print('Using device:', device)

if seed is None:
    seed = torch.seed()
torch.manual_seed(seed)

max_frames = sequence_settings['max_frames']
initial_fps = sequence_settings['initial_fps']

img_settings = config['img_settings']
model_type = img_settings['model_type']
width = img_settings['width']
height = img_settings['height']
model = img_settings['model']
initial_image = ""

#need to figure out if these should be fixed or configurable
img_gen_settings = {'model_config': f'checkpoints/{model}.yaml', 'vqgan_checkpoint':f'checkpoints/{model}.ckpt', 'size': [width, height]}

#process prompt
prompt_handler = PromptHandler(sequence_settings['mode'])
processed_sequence_settings = prompt_handler.handle(**sequence_settings)

if sequence_settings['mode'] == 'animation_2d':
    sequence_maker = Animator2D(model_type, img_gen_settings, device,  max_frames,initial_image, step_dir,save_all_iterations)
    sequence_maker.run(**processed_sequence_settings)
elif sequence_settings['mode'] == 'interpolation':
    sequence_maker = AnimatorInterpolate(model_type, img_gen_settings, device,  max_frames,initial_image, step_dir,save_all_iterations)
    sequence_maker.run(**processed_sequence_settings)

if do_super_res:
    upscaler = Upscaler(super_res_settings, device)
    upscaler.upscale_w_swinir(step_dir)
 
#export the video if slowmo is not done
if not(do_slowmo):
    if do_super_res:
        image_path = f'{step_dir}/super_res/*.png'
        video_name = os.path.join(exp_dir,"video_zoomed.mp4")
    else:
        image_path = f'{step_dir}/*.png'
        video_name = os.path.join(exp_dir,"video.mp4")
    export_to_ffmpeg(image_path,initial_fps, video_name)

if do_slowmo:
    video_name = os.path.join(exp_dir,'video_zoomed_slomo.mp4')
    if do_super_res:
        source_dir = os.path.join(step_dir,'super_res')
    else:
        source_dir = step_dir
    motion_slower = MotionSlower(slowmo_settings, device, batch_size=1)
    motion_slower.slomo(source_dir, video_name)

# merge the video and audio
if do_slowmo:
    generated_seconds = ((sequence_settings['max_frames']-2)*slowmo_settings['slowmo_factor'])/slowmo_settings['target_fps']
else:
    generated_seconds = (sequence_settings['max_frames']-2)/sequence_settings['initital_fps']
video_clip = VideoFileClip(video_name)
audio_clip = AudioFileClip('thoughtsarebeings_clip.wav')
audio_clip = audio_clip.subclip(t_start=0, t_end=generated_seconds)
final_clip = video_clip.set_audio(audio_clip)
final_clip.write_videofile(os.path.join(exp_dir,'video_with_audio.mp4'))