import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os, sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

BS = 4
print("batch size:", BS)
crop_size = 256

def tensorShow(tensors, titles=None):
    '''
    t:BCWH
    '''
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()

class LSRPD_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format=''):
        super(LSRPD_Dataset, self).__init__()
        self.size = size
        print(' train crop size', size)
        self.train = train
        self.format = format

        # Check if directories exist
        haze_dir = os.path.join(path, 'smoke')
        clear_dir = os.path.join(path, 'clean')
        if not os.path.exists(haze_dir):
            raise FileNotFoundError(f"Hazy image directory not found: {haze_dir}")
        if not os.path.exists(clear_dir):
            raise FileNotFoundError(f"Clear image directory not found: {clear_dir}")

        self.haze_imgs_dir = os.listdir(haze_dir)
        self.haze_imgs = [os.path.join(haze_dir, img) for img in self.haze_imgs_dir]
        self.clear_dir = clear_dir

    def __getitem__(self, index):
        try:
            haze = Image.open(self.haze_imgs[index])
        except Exception as e:
            raise RuntimeError(f"Error loading image {self.haze_imgs[index]}: {e}")

        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, len(self.haze_imgs) - 1)
                haze = Image.open(self.haze_imgs[index])

        img = self.haze_imgs[index]
        id = os.path.basename(img).split('_')[0]
        clear_name = id + self.format
        clear_path = os.path.join(self.clear_dir, clear_name)

        if not os.path.exists(clear_path):
            raise FileNotFoundError(f"Clear image not found: {clear_path}")

        clear = Image.open(clear_path)
        clear = tfs.CenterCrop(haze.size[::-1])(clear)

        if not isinstance(self.size, str):
            if haze.size[0] < self.size or haze.size[1] < self.size:
                raise ValueError(f"Image size {haze.size} is smaller than crop size {self.size}")
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(bool(rand_hor))(data)
            target = tfs.RandomHorizontalFlip(bool(rand_hor))(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)


pwd = os.getcwd()
path = os.path.abspath(os.path.join(pwd, './LSRPD'))  # Use absolute path for robustness

LSRPD_train_loader = DataLoader(
    dataset=LSRPD_Dataset(os.path.join(path, 'train'), train=True, size=crop_size),
    batch_size=BS,
    shuffle=True
)
LSRPD_test_loader = DataLoader(
    dataset=LSRPD_Dataset(os.path.join(path, 'test'), train=False, size='whole img'),
    batch_size=1,
    shuffle=False
)

if __name__ == "__main__":
    # Test the data loader
    for haze, clear in LSRPD_train_loader:
        print(f"Haze shape: {haze.shape}, Clear shape: {clear.shape}")

        break
