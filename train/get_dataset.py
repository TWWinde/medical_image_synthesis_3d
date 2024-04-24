import sys
sys.path.append('/misc/no_backups/s1449/medical_image_synthesis_3d')
sys.path.append('/misc/no_backups/s1449')
from dataset_ import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, SynthRAD2023Dataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(opt):

    if opt.dataset_name == 'SynthRAD2023':
        train_dataset = SynthRAD2023Dataset(
            root_dir=opt.root_dir)
        val_dataset = SynthRAD2023Dataset(
            root_dir=opt.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler
    raise ValueError(f'{opt.dataset_name} Dataset is not available')
