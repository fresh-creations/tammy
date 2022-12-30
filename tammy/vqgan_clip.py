import torch
from clip import clip
from torch import optim
from PIL import Image
import kornia.augmentation as K
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from torch import nn
from taming.models import cond_transformer, vqgan
import math
import numpy as np
import os
import imageio
from mega import Mega
from omegaconf import OmegaConf

def save_output(i, img, step_dir, suffix=None):
    filename = \
        f"{step_dir}/{i:06}{'_' + suffix if suffix else ''}.png"
    imageio.imwrite(filename, np.array(img))

def out_to_img(out):
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    return img

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        print(config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2]) 

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
replace_grad = ReplaceGrad.apply

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
 
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
 
 
clamp_with_grad = ClampWithGrad.apply


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))
 
 
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()
 
 
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]
 
 
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        self.noise_fac = 0.1
 
 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class ImgGenerator:

    def __init__(self,device,make_cutouts, img_gen_settings):
        model_config = img_gen_settings['model_config']
        vqgan_checkpoint = img_gen_settings['vqgan_checkpoint']
        size = img_gen_settings['size']
        self.device = device
        self.model = load_vqgan_model(model_config, vqgan_checkpoint).to(device)
        self.e_dim = self.model.quantize.e_dim #256
        f = 2**(self.model.decoder.num_resolutions - 1) #16
        self.make_cutouts = make_cutouts
        self.n_toks = self.model.quantize.n_e
        self.toksX, self.toksY = size[0] // f, size[1] // f
        self.sideX, self.sideY = self.toksX * f, self.toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    def synth(self, z):
        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    def im_to_tensor(self,img_0):
        return TF.to_tensor(img_0).unsqueeze(0) * 2 - 1

    def get_random(self):
        one_hot = F.one_hot(
            torch.randint(self.n_toks, [self.toksY * self.toksX], device=self.device), self.n_toks
        ).float()
        z = one_hot @ self.model.quantize.embedding.weight
        z = z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(0, 3, 1, 2)
        return z

    def encode(self,x):
        x = self.im_to_tensor(x).to(self.device)
        return self.model.encode(x)



class CLIP_wrapper:
    def __init__(self, device, make_cutouts, clip_settings) -> None:
        self.device = device
        self.make_cutouts = make_cutouts
        clip_model = clip_settings['clip_model']
        self.perceptor = clip.load(clip_model, device = device, jit=False)[0].eval().requires_grad_(False).to(self.device)

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])


    def encode_prompts(self,prompts, image_prompts, noise_prompt_seeds, noise_prompt_weights ):

        pMs = []

        #encode text prompt
        for prompt in prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))

        #encode image prompt
        for prompt in image_prompts:
            path, weight, stop = parse_prompt(prompt)
            sideX = 1
            sideY = 1
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))

        #add noise to prompt?
        for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(self.device))

        return pMs

    def encode_image(self,out):
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()
        return iii

class VQGAN_CLIP:
    def __init__(self, img_gen_settings, device) -> None:


        mega = Mega()
        m = mega.login()
        url = 'https://mega.nz/file/2CxihJYS#9crWU7POY9V2g6kbYZF1DQpfIxhG1RqSewzszKNCpfM'
        path = "./checkpoints"
        filename =  "vqgan_imagenet_f16_16384.yaml"
        ckpt_path = os.path.join(path,filename)
        checkpoint_downloaded = os.path.exists(ckpt_path)
        if not checkpoint_downloaded:
            print(f'downloading {filename}')
            m.download_url(url,path,filename)
        
        url = 'https://mega.nz/file/feoBhKYZ#874wbQplVucaUfo6hhpzW7AuJXd1mijyrAzxqFicp8U'
        path = "./checkpoints"
        filename =  "vqgan_imagenet_f16_16384.ckpt"
        ckpt_path = os.path.join(path,filename)
        checkpoint_downloaded = os.path.exists(ckpt_path)
        if not checkpoint_downloaded:
            print(f'downloading {filename}')
            m.download_url(url,path,filename)

        cut_out_settings = {'cut_size': 224, 'cutn':64, 'cut_pow': 1}
        clip_settings = {'clip_model': 'ViT-B/32'}
        train_settings = {'init_weight': 0., 'step_size': 0.1}

        self.make_cutouts = MakeCutouts(**cut_out_settings) 
        self.img_generator = ImgGenerator(device, self.make_cutouts, img_gen_settings) 
        self.clip = CLIP_wrapper(device,self.make_cutouts, clip_settings)
        self.step_size = train_settings['step_size']
        self.init_weight = train_settings['init_weight']

    def get_image(self, i, img, step_dir, prompts, image_prompts, noise_prompt_seeds, noise_prompt_weights, iterations_per_frame, save_all_iterations):

        if i == 0:
            z = self.img_generator.get_random()
        else:
            z, *_ = self.img_generator.encode(img)
        z_orig = z.clone()
        z.requires_grad_(True)
        
        pMs = self.clip.encode_prompts(prompts, image_prompts, noise_prompt_seeds, noise_prompt_weights)

        #save output image?
        
        #training loop for every frame
        for n in range(iterations_per_frame):
            if n == iterations_per_frame-1:
                save_flag = True
            else: 
                save_flag = save_all_iterations
            suffix = (str(n+1) if save_all_iterations else None)

            out, iii = self.forward(z)
            losses = self.train(iii, z, z_orig, pMs)
            

            img = out_to_img(out)
            if save_flag:
                save_output(i+1, img, step_dir,suffix=suffix)

            if iterations_per_frame == 0:
                save_output(i, img,step_dir)
            


    #do vector_quantize, VQGAN-decode and CLIP-image-encode
    def forward(self, z):
        out = self.img_generator.synth(z)
        iii = self.clip.encode_image(out)

        return out, iii

    def train(self, iii, z, z_orig, pMs ):

        opt = optim.Adam([z], lr=self.step_size)
        losses = []
        if self.init_weight:
            losses.append(F.mse_loss(z, z_orig) * self.init_weight / 2)

        for prompt in pMs:
            losses.append(prompt(iii))

        opt.zero_grad()
        lossAll = losses
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(self.img_generator.z_min).minimum(self.img_generator.z_max))

        return loss
