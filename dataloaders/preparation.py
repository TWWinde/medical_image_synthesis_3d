import os
import nibabel as nib
import numpy as np
import shutil
from sklearn.model_selection import train_test_split


def crop_center(img, new_x, new_y):
    x, y, z = img.shape
    start_x = x // 2 - new_x // 2
    start_y = y // 2 - new_y // 2
    return img[start_x:start_x + new_x, start_y:start_y + new_y, :]


def save_cropped(files, folder, crop_size):
    for file_path in files:

        img = nib.load(file_path)
        data = img.get_fdata()

        cropped_data = crop_center(data, *crop_size)

        cropped_img = nib.Nifti1Image(cropped_data, affine=img.affine)

        output_path = os.path.join(folder, os.path.basename(file_path))

        nib.save(cropped_img, output_path)


def process_images(source_folder, train_folder, test_folder, crop_size=(256, 256)):

    ct_train_folder = os.path.join(train_folder, 'ct')
    ct_test_folder = os.path.join(test_folder, 'ct')
    label_train_folder = os.path.join(train_folder, 'label')
    label_test_folder = os.path.join(test_folder, 'label')

    os.makedirs(os.path.join(train_folder, 'ct'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'label'), exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    ct_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('0001.nii.gz')]

    ct_train_files, ct_test_files = train_test_split(ct_files, test_size=0.1, random_state=42)
    label_train_files = [path.replace('0001.nii.gz', '0002.nii.gz') for path in ct_train_files]
    label_test_files = [path.replace('0001.nii.gz', '0002.nii.gz') for path in ct_test_files]

    save_cropped(ct_train_files, ct_train_folder, crop_size)
    save_cropped(ct_test_files, ct_test_folder, crop_size)
    save_cropped(label_train_files, label_train_folder, crop_size)
    save_cropped(label_test_files, label_test_folder, crop_size)


if __name__ == '__main__':

    source_folder = '/data/private/autoPET/imagesTr'
    train_folder = '/data/private/autoPET/autopet_3d/train'
    test_folder = '/data/private/autoPET/autopet_3d/test'

    process_images(source_folder, train_folder, test_folder)

