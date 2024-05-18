from torch.utils.data import Dataset
import torchio as tio
import os


import os

PREPROCESSING_RescaleIntensity = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1))
])

PREPROCESSING_CropOrPad = tio.Compose([
    tio.CropOrPad(target_shape=(256, 256, 32))
])

TRAIN_TRANSFORMS = tio.Compose([
    # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(0, 0, 3), translation=(4, 4, 0)),
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])


class AutoPET(Dataset):
    def __init__(self, opt, train_diffusion=False):
        super().__init__()
        self.opt = opt
        self.root_dir = opt.root_dir
        self.preprocessing_RescaleIntensity = PREPROCESSING_RescaleIntensity
        self.preprocessing_CropOrPad = PREPROCESSING_CropOrPad
        self.transforms = TRAIN_TRANSFORMS
        self.train_diffusion = train_diffusion
        if self.train_diffusion:
            self.ct_folder_names, self.label_folder_names = self.get_data_files()
        else:
            self.ct_folder_names = self.get_data_files()

    def get_data_files(self):
        mode = "test" if self.opt.phase == "test" else "train"
        if self.train_diffusion:
            label_subfolder_names = os.listdir(os.path.join(self.root_dir, mode, 'label'))
            label_folder_names = [os.path.join(
                self.root_dir, mode, 'label', subfolder) for subfolder in label_subfolder_names]
            ct_folder_names = [os.path.join(
                self.root_dir, mode, 'ct', subfolder.replace('0002.nii.gz', '0001.nii.gz')) for subfolder in
                label_subfolder_names]
            assert len(ct_folder_names) == len(label_folder_names)
            return ct_folder_names, label_folder_names
        else:
            ct_subfolder_names = os.listdir(os.path.join(self.root_dir, mode, 'ct'))
            ct_folder_names = [os.path.join(
                self.root_dir, mode, 'ct', subfolder) for subfolder in ct_subfolder_names]
            return ct_folder_names

    def __len__(self):
        return len(self.ct_folder_names)

    def __getitem__(self, idx: int):
        if self.train_diffusion:
            img = tio.ScalarImage(self.ct_folder_names[idx])
            label = tio.ScalarImage(self.label_folder_names[idx])
            img = self.preprocessing_CropOrPad(img)
            label = self.preprocessing_CropOrPad(label)
            img = self.preprocessing_RescaleIntensity(img)

            return {'ct_image': img.data.permute(0, -1, 1, 2), 'ct_label': label}
        else:
            img = tio.ScalarImage(self.ct_folder_names[idx])
            img = self.preprocessing_CropOrPad(img)
            img = self.preprocessing_RescaleIntensity(img)
            img = self.transforms(img)
            return {'data': img.data.permute(0, -1, 1, 2)}


