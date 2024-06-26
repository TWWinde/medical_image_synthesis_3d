import torch
import sys
sys.path.append('/misc/no_backups/d1502/medical_image_synthesis_3d')


def get_dataset_name(mode):

    if mode == "SynthRAD2023":
        return "SynthRAD2023"
    if mode == "AutoPET":
        return "AutoPET"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders." + dataset_name)

    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt)
    dataset_val = file.__dict__[dataset_name].__dict__[dataset_name](opt)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                                   drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False,
                                                 drop_last=False)

    return dataloader_train, dataloader_val