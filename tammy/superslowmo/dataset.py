import os
import os.path

from torch.utils.data import Dataset
from skimage import io

class SloMoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        img_name_1 = os.path.join(self.root_dir,f'{1:06d}.png')
        frame = io.imread(img_name_1)
        self.origDim = (frame.shape[1],frame.shape[0])
        self.dim = int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32

    def __len__(self):
        return len(os.listdir(self.root_dir))-1

    def __getitem__(self, idx):
        img_name_1 = os.path.join(self.root_dir,f'{idx+1:06d}.png')
        img_name_2 = os.path.join(self.root_dir,f'{idx+2:06d}.png')
        frame_1 = io.imread(img_name_1)
        frame_2 = io.imread(img_name_2)

        if self.transform:
            frame_1 = self.transform(frame_1)
            frame_2 = self.transform(frame_2)

        return frame_1, frame_2