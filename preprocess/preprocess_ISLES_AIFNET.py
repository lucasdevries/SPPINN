import os
import numpy as np
import SimpleITK as sitk
import glob
import pickle
from tqdm import tqdm
import nibabel as nib
import argparse
import re

import matplotlib.pyplot as plt


def bilateral_filter(img: sitk.Image):
    filter = sitk.BilateralImageFilter()
    filter.SetRangeSigma(20)
    filter.SetDomainSigma(3)
    return filter.Execute(img)


def split_4d_sitk_to_3d(img: sitk):
    num_volumes = img.GetSize()[-1]
    ctp_volume_list = []
    for i in range(num_volumes):
        image_3d = sitk.Extract(img, (img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], 0),
                                (0, 0, 0, i))
        ctp_volume_list.append(image_3d)
    return ctp_volume_list


def new_img_size(img, new_spacing):
    if isinstance(new_spacing, int) or isinstance(new_spacing, float):
        new_spacing = [new_spacing, new_spacing, new_spacing]
    new_size = []
    for ix, sp in enumerate(new_spacing):
        new_size.append(int(np.ceil(img.GetSize()[ix] * img.GetSpacing()[ix] / sp)))
    return new_size


def Resample_img(img, new_spacing, interpolator=sitk.sitkLinear):
    # new_spacing should be in sitk order x,y,z (np order: z,y,x)
    if isinstance(new_spacing, int) or isinstance(new_spacing, float):
        new_spacing = [new_spacing, new_spacing, new_spacing]
    # https://github.com/SimpleITK/SimpleITK/issues/561
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator = interpolator
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    new_size = new_img_size(img, new_spacing)
    resample.SetSize(new_size)
    img = resample.Execute(img)
    img = sitk.Cast(img, sitk.sitkInt32)
    return img


def process_PID(PID: str, dataset='TRAINING'):
    dataset_path = os.path.join('..', 'data', 'ISLES2018', dataset)
    patient_path = os.path.join(dataset_path, PID)
    # get the files
    ctp = glob.glob(os.path.join(patient_path, "*CT_4DPWI*", "*CT_4DPWI*.nii"))[0]
    # read the files
    ctp = sitk.ReadImage(ctp)
    ctp_frames = split_4d_sitk_to_3d(ctp)

    # apply bilateral filter
    ctp_frames = [bilateral_filter(img) for img in tqdm(ctp_frames)]

    cbf = sitk.ReadImage(glob.glob(os.path.join(patient_path, "*CT_CBF*", "*CT_CBF*.nii"))[0])
    cbv = sitk.ReadImage(glob.glob(os.path.join(patient_path, "*CT_CBV*", "*CT_CBV*.nii"))[0])
    mtt = sitk.ReadImage(glob.glob(os.path.join(patient_path, "*CT_MTT*", "*CT_MTT*.nii"))[0])
    tmax = sitk.ReadImage(glob.glob(os.path.join(patient_path, "*CT_Tmax*", "*CT_Tmax*.nii"))[0])
    if dataset == 'TRAINING':
        core = sitk.ReadImage(glob.glob(os.path.join(patient_path, "*OT*", "*OT*.nii"))[0])

    #cast
    ctp_frames = [sitk.Cast(img, sitk.sitkInt32) for img in ctp_frames]
    cbf = sitk.Cast(cbf, sitk.sitkInt32)
    cbv = sitk.Cast(cbv, sitk.sitkInt32)
    mtt = sitk.Cast(mtt, sitk.sitkInt32)
    tmax = sitk.Cast(tmax, sitk.sitkInt32)


    # get current spacing
    spacing = cbf.GetSpacing()
    # new spacing
    new_spacing = [spacing[0] / 2, spacing[1] / 2, spacing[2]]
    # resample
    # ctp_frames = [Resample_img(img, new_spacing=new_spacing) for img in ctp_frames]
    # cbf = Resample_img(cbf, new_spacing=new_spacing)
    # cbv = Resample_img(cbv, new_spacing=new_spacing)
    # mtt = Resample_img(mtt, new_spacing=new_spacing)
    # tmax = Resample_img(tmax, new_spacing=new_spacing)

    brainmask = np.zeros_like(sitk.GetArrayFromImage(cbf))
    brainmask[sitk.GetArrayFromImage(cbf) > 0] = 1
    brainmask[sitk.GetArrayFromImage(cbv) > 0] = 1
    brainmask[sitk.GetArrayFromImage(mtt) > 0] = 1
    brainmask[sitk.GetArrayFromImage(tmax) > 0] = 1
    brainmask = sitk.GetImageFromArray(brainmask)
    brainmask.CopyInformation(cbf)

    # if dataset == 'TRAINING':
    #     core = Resample_img(core, new_spacing=new_spacing, interpolator=sitk.sitkNearestNeighbor)

    # save the files
    save_path = os.path.join('..', 'data', 'ISLES2018', f'{dataset}_processed_2', PID)
    os.makedirs(save_path, exist_ok=True)
    for i, img in enumerate(ctp_frames):
        sitk.WriteImage(img, os.path.join(save_path, f"CTP_{str(i).zfill(2)}.nii.gz"))
    sitk.WriteImage(cbf, os.path.join(save_path, "CBF.nii.gz"))
    sitk.WriteImage(cbv, os.path.join(save_path, "CBV.nii.gz"))
    sitk.WriteImage(mtt, os.path.join(save_path, "MTT.nii.gz"))
    sitk.WriteImage(tmax, os.path.join(save_path, "Tmax.nii.gz"))
    sitk.WriteImage(brainmask, os.path.join(save_path, "brainmask.nii.gz"))
    if dataset == 'TRAINING':
        sitk.WriteImage(core, os.path.join(save_path, "core.nii.gz"))

    # get the AIF from the AIFNET data
    gt_path_2 = os.path.join('..', 'data', 'AIFNET', 'annotations_rater_II')
    # Open voxel coordinate annotations
    with open(gt_path_2, "rb") as f:
        annotations_2 = pickle.load(f)[0]

    gt_path_1 = os.path.join('..', 'data', 'AIFNET', 'annotations_rater_I')
    # Open voxel coordinate annotations
    with open(gt_path_1, "rb") as f:
        annotations_1 = pickle.load(f)[0]

    case_nr = int(PID.split('_')[1])
    if dataset == 'TRAINING':
        case_annotations_1 = annotations_1[case_nr - 1]
    else:
        case_annotations_1 = annotations_1[case_nr + 93]
    if dataset == 'TRAINING':
        case_annotations_2 = annotations_2[case_nr - 1]
    else:
        case_annotations_2 = annotations_2[case_nr + 93]

    # Extract vascular functions folowing the AIFNET paper code in data/AIFNET/original.py
    ctp = glob.glob(os.path.join(patient_path, "*CT_4DPWI*", "*CT_4DPWI*.nii"))[0]
    myCTP = nib.load(ctp)
    myCTP = myCTP.get_fdata()

    aif_1 = myCTP[case_annotations_1['aif_voxel_coordinate']]
    vof_1 = myCTP[case_annotations_1['vof_voxel_coordinate']]
    aif_2 = myCTP[case_annotations_2['aif_voxel_coordinate']]
    vof_2 = myCTP[case_annotations_2['vof_voxel_coordinate']]

    aif = np.mean([aif_1, aif_2], axis=0).astype(np.float32)
    vof = np.mean([vof_1, vof_2], axis=0).astype(np.float32)
    # Save vascular functions
    np.save(os.path.join(save_path, 'AIF.npy'), aif)
    np.save(os.path.join(save_path, 'VOF.npy'), vof)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='all', help='process a specific case')
    parser.add_argument('--dataset', type=str, default='TRAINING', help='process the dataset TRAINING or TESTING')
    args = parser.parse_args()

    pattern = re.compile("^case_\d{1,2}$")
    if args.case == 'all':
        if args.dataset == 'TRAINING':
            for case in tqdm(range(1, 95)):
                process_PID(f'case_{case}', dataset='TRAINING')
        else:
            for case in tqdm(range(1, 63)):
                process_PID(f'case_{case}', dataset='TESTING')
    elif pattern.match(args.case):
        if args.dataset == 'TRAINING':
            process_PID(args.case, dataset='TRAINING')
        else:
            process_PID(args.case, dataset='TESTING')
    else:
        raise ValueError('Please specify either a correct case number or "all"')
