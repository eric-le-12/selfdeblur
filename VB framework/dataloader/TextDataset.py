from torch.utils.data import Dataset
import os
from PIL import Image


class Text(Dataset):
    def __init__(self, path, img_transform=None, blur_transform=None):
        self.path = path
        self.no_images = len(os.listdir(self.path)) // 3
        self.img_transform = img_transform
        self.blur_transform = blur_transform

    def __len__(self):
        return self.no_images

    def __getitem__(self, idx):

        assert 0 <= idx <= self.no_images - 1, "Invalid index"

        image_prefix = (7 - len(str(idx))) * "0" + str(idx)
        blur_img_path = os.path.join(self.path, image_prefix + "_blur.png")
        blur_path = os.path.join(self.path, image_prefix + "_psf.png")
        clean_img_path = os.path.join(self.path, image_prefix + "_orig.png")

        blur_kernel = Image.open(blur_path)
        blur_img = Image.open(blur_img_path)
        clean_img = Image.open(clean_img_path)

        if self.img_transform:
            blur_img = self.img_transform(blur_img)
            clean_img = self.img_transform(clean_img)
        if self.blur_transform:
            blur_kernel = self.blur_transform(blur_kernel)

        return blur_kernel, blur_img, clean_img
