from tammy.utils import empty_cuda
import os
import torch
import cv2
import wget
import numpy as np
from tqdm import tqdm
from tammy.upscaling.network_swinir import SwinIR as net
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.io import read_image


TRANSFORM_IMG = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    ])

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))-1

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{(idx+1):06d}.png')
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image


class Upscaler:
    def __init__(self, super_res_settings, device) -> None:
        self.scale = super_res_settings['upscale_factor']
        self.batch_size = super_res_settings['batch_size']
        self.device = device
        url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
        path = "./checkpoints"
        checkpoint_downloaded = os.path.exists(os.path.join(path,"003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"))
        if not checkpoint_downloaded:
            print('downloading super-resolution model: SwinIR')
            wget.download(url,out = path)

    def upscale_w_swinir(self,step_dir):
        print('Starting Super-Resolution')
        empty_cuda()
        super_res_dir = os.path.join(step_dir,'super_res')
        os.mkdir(super_res_dir)

        model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

        param_key_g = 'params_ema'
        pretrained_model = torch.load('checkpoints/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        model.eval()
        model = model.to(self.device)

        window_size = 8
        dataset = CustomImageDataset(img_dir=step_dir, transform=TRANSFORM_IMG)
        dataloader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,  num_workers=2)

        for idx, img_lq in enumerate((pbar := tqdm(dataloader))):
            img_lq = img_lq.to(self.device)
            pbar.set_description(f'upscaling frames')
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = model(img_lq)
                output = output[..., :h_old * self.scale, :w_old * self.scale]

            # save image
            bs = output.shape[0]
            for img_idx in range(bs):
                idx_save = self.batch_size*idx+img_idx
                output_img = output[img_idx].unsqueeze(0)
                output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if output_img.ndim == 3:
                    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                output_img = (output_img * 255.0).round().astype(np.uint8)  # float32 to uint8
                imgname = f'{(idx_save+1):06d}'
                output_path = os.path.join(super_res_dir,imgname+'_SwinIR.png')
                cv2.imwrite(output_path, output_img)
        print('Super-Resolution done')