import os
from shutil import rmtree
from PIL import Image
import torch
import torchvision.transforms as transforms
from tammy.superslowmo import model
import gdown
from tammy.superslowmo import dataloader
from tqdm import tqdm
from tammy.utils import empty_cuda, export_to_ffmpeg

from tammy.superslowmo.dataloader import SloMoDataset

class MotionSlower():
    def __init__(self, slowmo_settings, device, batch_size = 1) -> None:
        self.target_fps = slowmo_settings['target_fps']
        self.slowmo_factor = slowmo_settings['slowmo_factor']
        self.batch_size = batch_size
        self.device = device

        url = "https://drive.google.com/uc?id=1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF"
        path = "./checkpoints"
        checkpoint_path = os.path.join(path,"SuperSloMo.ckpt")
        checkpoint_downloaded = os.path.exists(checkpoint_path)
        if not checkpoint_downloaded:
            print('downloading SuperSloMo.ckpt')
            gdown.download(url,checkpoint_path)

    def slomo(self,input_path,video_path):
        empty_cuda()
        print("Starting slowmo")
        pretrained_model = 'checkpoints/SuperSloMo.ckpt'
        # Check if arguments are okay
        extractionDir = ".tmpSuperSloMo"

        if os.path.isdir(extractionDir):
            rmtree(extractionDir)

        os.mkdir(extractionDir)

        outputPath = os.path.join(extractionDir, "output")
        os.mkdir(outputPath)

        # Initialize transforms

        mean = [0.429, 0.431, 0.397]
        std  = [1, 1, 1]
        normalize = transforms.Normalize(mean=mean,
                                        std=std)

        negmean = [x * -1 for x in mean]
        revNormalize = transforms.Normalize(mean=negmean, std=std)

        # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
        # - Removed per channel mean subtraction for CPU.
        if (self.device == "cpu"):
            transform = transforms.Compose([transforms.ToTensor()])
            TP = transforms.Compose([transforms.ToPILImage()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
            TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

        # Load data
        dataset = SloMoDataset(root_dir=input_path,transform=transform)
        slomoloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        #videoFrames = dataloader.Video(root=input_path, transform=transform)
        #videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        flowComp = model.UNet(6, 4)
        flowComp.to(self.device)
        for param in flowComp.parameters():
            param.requires_grad = False
        ArbTimeFlowIntrp = model.UNet(20, 5)
        ArbTimeFlowIntrp.to(self.device)
        for param in ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        #print('dim 1', videoFrames.dim[0])
        #print('dim 2', videoFrames.dim[1])
        #TODO: READ this number
        dim_1 = dataset.dim[0]
        dim_2 = dataset.dim[1]
        orig_dim = dataset.origDim
        flowBackWarp = model.backWarp(dim_1, dim_2, self.device)
        flowBackWarp = flowBackWarp.to(self.device)

        dict1 = torch.load(pretrained_model, map_location='cpu')
        ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
        flowComp.load_state_dict(dict1['state_dictFC'])

        # Interpolate frames
        frameCounter = 1

        with torch.no_grad():
            for _, (frame0, frame1) in enumerate(slomoloader):
            #for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

                I0 = frame0.to(self.device)
                I1 = frame1.to(self.device)

                flowOut = flowComp(torch.cat((I0, I1), dim=1))
                F_0_1 = flowOut[:,:2,:,:]
                F_1_0 = flowOut[:,2:,:,:]

                # Save reference frames in output folder
                for batchIndex in range(self.batch_size):
                    (TP(frame0[batchIndex].detach())).resize(orig_dim, Image.BILINEAR).save(os.path.join(outputPath, f'{frameCounter + self.slowmo_factor * batchIndex:06d}.png'))
                frameCounter += 1

                # Generate intermediate frames
                for intermediateIndex in range(1, self.slowmo_factor):
                    t = float(intermediateIndex) / self.slowmo_factor
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                    g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                    intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1   = 1 - V_t_0

                    g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                    g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(self.batch_size):
                        (TP(Ft_p[batchIndex].cpu().detach())).resize(orig_dim, Image.BILINEAR).save(os.path.join(outputPath, f'{frameCounter + self.slowmo_factor * batchIndex:06d}.png'))
                    frameCounter += 1

                # Set counter accounting for batching of frames
                frameCounter += self.slowmo_factor * (self.batch_size - 1)

        export_to_ffmpeg(f'{outputPath}/*.png',self.target_fps, video_path)
        
        # Remove temporary files
        rmtree(extractionDir)
        print("Slow-mo is ready")
