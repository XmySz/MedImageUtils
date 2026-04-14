import SimpleITK as sitk


def dcm2nii(dcms_dir, nii_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_dir)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    image3 = sitk.GetImageFromArray(image_array)
    image3.CopyInformation(image2)
    sitk.WriteImage(image3, nii_path)


if __name__ == '__main__':
    pass
