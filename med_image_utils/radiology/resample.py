import nibabel as nib
import numpy as np
import scipy.ndimage
import os
from tqdm import tqdm


def resample_nifti_to_spacing(image_path, label_path, image_interp_order=3, label_interp_order=0):
    img_nib = nib.load(image_path)
    img_data = img_nib.get_fdata()
    original_affine = img_nib.affine
    original_spacing = np.array(img_nib.header.get_zooms()[:3])

    lbl_nib = nib.load(label_path)
    lbl_data = lbl_nib.get_fdata()

    zoom_factors = original_spacing / np.array(target_spacing)

    resampled_img_data = scipy.ndimage.zoom(img_data, zoom_factors, order=image_interp_order, mode='nearest')
    resampled_lbl_data = scipy.ndimage.zoom(lbl_data, zoom_factors, order=label_interp_order, mode='nearest')

    if np.issubdtype(lbl_data.dtype, np.integer):
        resampled_lbl_data = np.round(resampled_lbl_data).astype(lbl_data.dtype)
    else:
        pass

    new_affine = original_affine.copy()

    for i in range(3):
        original_axis_length = np.sqrt(np.sum(original_affine[:3, i] ** 2))
        if original_axis_length > 1e-9:
            new_affine[:3, i] = (original_affine[:3, i] / original_axis_length) * target_spacing[i]

    new_img_header = img_nib.header.copy()
    new_img_header.set_zooms(tuple(target_spacing) + img_nib.header.get_zooms()[3:])

    new_lbl_header = lbl_nib.header.copy()
    new_lbl_header.set_zooms(tuple(target_spacing) + lbl_nib.header.get_zooms()[3:])

    resampled_img_nib = nib.Nifti1Image(resampled_img_data, new_affine, new_img_header)
    resampled_lbl_nib = nib.Nifti1Image(resampled_lbl_data, new_affine, new_lbl_header)

    nib.save(resampled_img_nib, image_path)
    nib.save(resampled_lbl_nib, label_path)


def resample_nifti_dataset_inplace(image_folder, label_folder, target_spacing):
    image_files = sorted([f for f in os.listdir(image_folder)])

    for img_name in tqdm(image_files, desc='Resampling dataset'):
        image_path = os.path.join(image_folder, img_name)
        label_path = os.path.join(label_folder, img_name.replace('_0000', ''))

        resample_nifti_to_spacing(image_path, label_path)


if __name__ == '__main__':
    image_folder = r"D:\Data\OvarianCancer\Datasets\CA_hos_Label\T2\Images"
    label_folder = r"D:\Data\OvarianCancer\Datasets\CA_hos_Label\T2\Labels"

    target_spacing = (1.5, 1.5, 1.5)

    resample_nifti_dataset_inplace(image_folder, label_folder, target_spacing)
