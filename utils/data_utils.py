import numpy as np
import SimpleITK as sitk
from einops.einops import rearrange, repeat
from scipy.ndimage import gaussian_filter, convolve
from skimage.restoration import denoise_bilateral
from skimage.filters import gaussian
import torch
import os, glob
import matplotlib.pyplot as plt
import scipy.io
import wandb
from scipy.integrate import simpson
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def load_phantom_data(gaussian_filter_type, sd=2.5,
              folder=r'data/DigitalPhantomCT',
              cbv_ml=5, simulation_method=2,
              temporal_smoothing=True,
              save_nifti=True,
              baseline_zero=True):
    print("Reading Dicom directory:", folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_data = sitk.GetArrayFromImage(image)
    # values: AIF/VOF, Exp R(t) for CBV 1-5, Lin R(t) for CBV 1-5, Box R(t) for CBV 1-5,
    image_data = rearrange(image_data, '(t values) h w -> values t h w', t=30)
    image_data = image_data.astype(np.float32)
    time = np.array([float(x) for x in range(0, 60, 2)])
    vof_location = (410, 247, 16)  # start, start, size
    vof_data = image_data[0,
               :,
               vof_location[0]:vof_location[0] + vof_location[2],
               vof_location[1]:vof_location[1] + vof_location[2]]
    vof_data = np.mean(vof_data, axis=(1, 2))

    aif_location = (123, 251, 8)  # start, start, size
    aif_data = image_data[0,
               :,
               aif_location[0]:aif_location[0] + aif_location[2],
               aif_location[1]:aif_location[1] + aif_location[2]]
    aif_data = np.mean(aif_data, axis=(1, 2))

    # Correct aif for partial volume effect
    vof_baseline = np.mean(vof_data[:4])
    aif_baseline = np.mean(aif_data[:4])
    aif_wo_baseline = aif_data - aif_baseline
    vof_wo_baseline = vof_data - vof_baseline
    cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
    cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
    ratio = cumsum_vof / cumsum_aif
    aif_data = aif_wo_baseline * ratio + aif_baseline

    if baseline_zero:
        aif_baseline = np.mean(aif_data[:4])
        aif_data = aif_data - aif_baseline
        tac_baseline = np.expand_dims(np.mean(image_data[:, :4, ...], axis=1), axis=1)
        image_data = image_data - tac_baseline

    simulated_data_size = 32 * 7
    scan_center = image_data.shape[-1] // 2
    simulated_data_start = scan_center - simulated_data_size // 2
    simulated_data_end = scan_center + simulated_data_size // 2
    if gaussian_filter_type:
        perfusion_data = image_data[1:, :,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
        perfusion_data = apply_gaussian_filter(gaussian_filter_type, perfusion_data.copy(), sd=sd)

    else:
        perfusion_data = image_data[1:, :,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
    perfusion_data = perfusion_data.astype(np.float32)

    if save_nifti:
        scipy.io.savemat(os.path.join('data', 'aif_data.mat'), {'aif': aif_data})
        perfusion_data_nii = rearrange(perfusion_data, 'c t h w -> h w c t')
        perfusion_data_nii = sitk.GetImageFromArray(perfusion_data_nii)
        sitk.WriteImage(perfusion_data_nii, os.path.join('data', 'NLR_image_data_sd_{}.nii'.format(sd)))

    if temporal_smoothing:
        k = np.array([0.25, 0.5, 0.25])
        aif_data = convolve(aif_data, k, mode='nearest')
        k = k.reshape(1, 3, 1, 1)
        perfusion_data = convolve(perfusion_data, k, mode='nearest')

    perfusion_values = np.empty([5, 7, 7, 4])
    cbv = [1, 2, 3, 4, 5]  # in ml / 100g
    mtt_s = [24.0, 12.0, 8.0, 6.0, 4.8, 4.0, 3.42857143]  # in seconds
    mtt_m = [t / 60 for t in mtt_s]  # in minutes
    delay = [0., 0.5, 1., 1.5, 2., 2.5, 3.]  # in seconds

    for ix, i in enumerate(cbv):
        for jx, j in enumerate(delay):
            for kx, k in enumerate(mtt_m):
                # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
                values = np.array([i, j, k, i / k])
                perfusion_values[ix, jx, kx] = values
    perfusion_values = repeat(perfusion_values, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    perfusion_values_dict = {'cbf': perfusion_values[simulation_method, cbv_ml-1, ..., 3],
                             'delay': perfusion_values[simulation_method, cbv_ml-1, ..., 1],
                             'cbv': perfusion_values[simulation_method, cbv_ml-1, ..., 0],
                             'mtt': perfusion_values[simulation_method, cbv_ml-1, ..., 2] * 60}
    perfusion_values_dict['tmax'] = perfusion_values_dict['delay'] + 0.5 * perfusion_values_dict['mtt']
    perfusion_data = rearrange(perfusion_data, '(n cbv) t h w -> n cbv h w t', n=3)

    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time,
                 'curves': perfusion_data[simulation_method:simulation_method + 1, cbv_ml-1:cbv_ml, :, :, :],
                 'perfusion_values': perfusion_values[simulation_method:simulation_method + 1, cbv_ml-1:cbv_ml,
                                     :, :, :]}

    data_dict = normalize_data(data_dict)
    data_dict = get_coll_points(data_dict)
    data_dict = get_tensors(data_dict)

    data_dict['time'] = np.tile(
        data_dict['time'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, 224, 224,1),
    ).astype(np.float32)
    data_dict['time_inference_highres'] = np.tile(
        data_dict['time_inference_highres'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, 224, 224,1),
    ).astype(np.float32)
    # create meshes
    data_dict = create_mesh(data_dict)
    data_dict = get_tensors(data_dict)
    data_dict['perfusion_values_dict'] = perfusion_values_dict

    return data_dict

def load_ISLES_data(folder=r'D:/SPPINN/data/ISLES2018',
                    case='case_1',
                    temporal_smoothing=True,
                    dataset = 'TRAINING',
                    baseline_zero=True,):
    image_data_path = os.path.join(folder, f'{dataset}_processed_2', f'{case}')
    frames = sorted(glob.glob(os.path.join(image_data_path, 'CTP_[0-9][0-9].nii.gz')))
    image_data_dict = read_nii_folder(frames)

    aif_data = np.load(os.path.join(image_data_path, rf'aif_isles.npy'))
    vof_data = np.load(os.path.join(image_data_path, rf'vof_isles.npy'))


    if dataset == 'TRAINING':
        dwi_segmentation = os.path.join(image_data_path, 'core.nii.gz')
        dwi_segmentation = sitk.ReadImage(dwi_segmentation)
    else:
        dwi_segmentation = None
    # load image data

    space = image_data_dict['spacing']
    # load time matrix
    time_data = np.arange(0, image_data_dict['array'].shape[0], 1).astype(np.float32)

    # load brainmask
    brainmask = os.path.join(image_data_path, 'brainmask.nii.gz')
    brainmask_data = sitk.GetArrayFromImage(sitk.ReadImage(brainmask))

    aif_time_data = time_data
    vof_time_data = time_data
    if temporal_smoothing:
        print('using temporal smoothing on aif')
        k = np.array([0.25, 0.5, 0.25])
        aif_data = convolve(aif_data.copy(), k, mode='nearest')
        vof_data = convolve(vof_data.copy(), k, mode='nearest')

    vof_baseline = np.mean(vof_data[:4])
    aif_baseline = np.mean(aif_data[:4])
    aif_wo_baseline = aif_data - aif_baseline
    vof_wo_baseline = vof_data - vof_baseline
    cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
    cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
    ratio = cumsum_vof / cumsum_aif
    aif_data = aif_wo_baseline * ratio + aif_baseline

    brainmask_data = np.expand_dims(brainmask_data, axis=0)
    image_data_dict['array'] = np.multiply(image_data_dict['array'], brainmask_data)
    image_data_dict['mip'] = np.max(image_data_dict['array'], axis=0)

    if baseline_zero:
        tac_baseline = np.mean(image_data_dict['array'][:4], axis=0, keepdims=True)
        image_data_dict['array'] = image_data_dict['array'] - tac_baseline

    complete_mask = brainmask_data[0]

    # if temporal_smoothing:
    #     print('using temporal smoothing on tacs')
    #     k = np.array([0.25, 0.5, 0.25])
    #     k = k.reshape(3, 1, 1, 1)
    #     image_data_dict['array'] = convolve(image_data_dict['array'].copy(), k, mode='nearest')

    # If smoothing, apply here
    image_data_dict['array'] = image_data_dict['array'].astype(np.float32)
    image_data_dict['array'] = rearrange(image_data_dict['array'], 't d h w -> d h w t')

    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time_data,
                 'curves': image_data_dict['array'],
                 'brainmask': brainmask_data,
                 'mip': image_data_dict['mip'],
                 'mask': complete_mask,
                 }
    x_dim = image_data_dict['array'].shape[1]
    # create meshes
    data_dict = normalize_data(data_dict)
    data_dict = get_coll_points(data_dict)
    data_dict['time'] = repeat(data_dict['time'], 't -> d t', d=brainmask_data.shape[1])
    data_dict['time'] = np.tile(
        data_dict['time'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, x_dim, x_dim, 1),
    ).astype(np.float32)


    data_dict['time_inference_highres'] = np.tile(
        data_dict['time_inference_highres'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, x_dim, x_dim, 1),
    ).astype(np.float32)
    data_dict = create_mesh_amc(data_dict)
    data_dict = get_tensors(data_dict)
    data_dict['aif_time'] = data_dict['time'][0]

    cbf = os.path.join(image_data_path, 'CBF.nii.gz')
    data_dict['cbf'] = sitk.ReadImage(cbf)

    if dataset == 'TRAINING':
        data_dict['dwi_segmentation'] = dwi_segmentation
    return data_dict

def create_mesh_amc(data_dict):
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0, data_dict['time'].shape[2] - 1, num=data_dict['time'].shape[2]),
        np.linspace(0, data_dict['time'].shape[3] - 1, num=data_dict['time'].shape[3]),
    )
    # mesh_x_hr, mesh_y_hr = np.meshgrid(
    #     np.linspace(0, data_dict['time'].shape[2] - 1, num=10*data_dict['time'].shape[2]),
    #     np.linspace(0, data_dict['time'].shape[3] - 1, num=10*data_dict['time'].shape[3]),
    # )

    mesh_data = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (data_dict['time'].shape[0],data_dict['time'].shape[1], 1, 1, 1)
    ).astype(np.float32)

    mesh_data_xy = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (1, 1, 1)
    ).astype(np.float32)

    # mesh_data_xy_hr = np.tile(
    #     np.stack((mesh_x, mesh_y), axis=-1), (1, 1, 1)
    # ).astype(np.float32)

    mesh_data_hr = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (data_dict['time_inference_highres'].shape[0], 1, 1, 1)
    ).astype(np.float32)

    data_dict['mesh_mean'] = mesh_data.mean()
    data_dict['mesh_std'] = mesh_data.std()
    data_dict['mesh_max'] = mesh_data.max()
    data_dict['mesh_min'] = mesh_data.min()

    data_dict['indices'] = mesh_data

    mesh_data_hr = (mesh_data_hr - data_dict['mesh_mean']) / data_dict['mesh_std']
    mesh_data_xy = (mesh_data_xy - data_dict['mesh_mean']) / data_dict['mesh_std']
    # mesh_data_xy_hr = (mesh_data_xy_hr - data_dict['mesh_mean']) / data_dict['mesh_std']
    mesh_data = (mesh_data - data_dict['mesh_mean']) / data_dict['mesh_std']

    data_dict['coordinates_highres'] = np.concatenate([data_dict['time_inference_highres'], mesh_data_hr], axis=3)

    data_dict['coordinates'] = np.concatenate([data_dict['time'], mesh_data], axis=4)
    data_dict['coordinates_xy_only'] = mesh_data_xy
    # data_dict['coordinates_xy_only_hr'] = mesh_data_xy_hr
    data_dict['coordinates'] = rearrange(data_dict['coordinates'], 'dim1 t x y vals -> dim1 x y t vals')
    data_dict['coordinates_highres'] = rearrange(data_dict['coordinates_highres'],'t x y vals -> x y t vals')

    return data_dict
def create_mesh(data_dict):
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0, data_dict['time'].shape[1] - 1, num=data_dict['time'].shape[1]),
        np.linspace(0, data_dict['time'].shape[2] - 1, num=data_dict['time'].shape[2]),
    )
    mesh_data = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (len(data_dict['time']), 1, 1, 1)
    ).astype(np.float32)

    mesh_data_hr = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (len(data_dict['time_inference_highres']), 1, 1,1)
    ).astype(np.float32)

    mesh_data_xy = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (1, 1, 1)
    ).astype(np.float32)

    data_dict['mesh_mean'] = mesh_data.mean()
    data_dict['mesh_std'] = mesh_data.std()
    data_dict['mesh_max'] = mesh_data.max()
    data_dict['mesh_min'] = mesh_data.min()
    data_dict['indices'] = mesh_data

    mesh_data_hr = (mesh_data_hr - mesh_data.mean()) / mesh_data.std()
    mesh_data_xy = (mesh_data_xy - mesh_data.mean()) / mesh_data.std()
    mesh_data = (mesh_data - mesh_data.mean()) / mesh_data.std()

    data_dict['coordinates'] = np.concatenate([data_dict['time'], mesh_data], axis=3)[np.newaxis, np.newaxis, ...]
    data_dict['coordinates_xy_only'] = mesh_data_xy[np.newaxis, np.newaxis, ...]
    data_dict['time_xy_highres'] = np.concatenate([data_dict['time_inference_highres'], mesh_data_hr], axis=3)[np.newaxis, np.newaxis, ...]
    data_dict['coordinates'] = rearrange(data_dict['coordinates'], 'dim1 dim2 t x y vals -> dim1 dim2 x y t vals')
    data_dict['time_xy_highres'] = rearrange(data_dict['time_xy_highres'],'dim1 dim2 t x y vals -> dim1 dim2 x y t vals')

    return data_dict


def normalize_data(data_dict):
    # input normalization
    data_dict['std_t'] = data_dict['time'].std()
    data_dict['mean_t'] = data_dict['time'].mean()
    data_dict['time'] = (data_dict['time'] - data_dict['time'].mean()) / data_dict['time'].std()
    data_dict['time_inference_highres'] = np.array([float(x) for x in np.arange(np.min(data_dict['time']), np.max(data_dict['time']) + 0.06, 0.06)])
    # output normalization
    max_ = data_dict['aif'].max()
    data_dict['aif'] /= max_
    data_dict['vof'] /= max_
    data_dict['curves'] /= max_
    return data_dict


def get_coll_points(data_dict):
    data_dict['coll_points'] = np.random.uniform(
        np.min(data_dict['time']), np.max(data_dict['time']), len(data_dict['time']) * 5 * 10 * 3
    ).astype(np.float32)
    data_dict['bound'] = np.array([np.min(data_dict['time'])])
    data_dict['coll_points_max'] = np.max(data_dict['coll_points'])
    data_dict['coll_points_min'] = np.min(data_dict['coll_points'])

    return data_dict

def get_tensors(data_dict):
    for key in data_dict.keys():
        data_dict[key] = torch.as_tensor(data_dict[key], dtype=torch.float32)
    return data_dict

def apply_gaussian_filter(type, array, sd):
    truncate = np.ceil(2 * sd) / sd if sd != 0 else 0
    if len(array.shape) == 4:
        if type == 'gauss_spatiotemporal':
            return gaussian(array, sigma=(0, sd, sd, sd), mode='nearest', truncate=truncate)
        elif type == 'gauss_spatial':
            return gaussian(array, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
        else:
            raise NotImplementedError('Gaussian filter variant not implemented.')

    if len(array.shape) == 3:
        if type == 'gauss_spatiotemporal':
            return gaussian(array, sigma=(sd, sd, sd), mode='nearest', truncate=truncate)
        elif type == 'gauss_spatial':
            return gaussian(array, sigma=(0, sd, sd), mode='nearest', truncate=truncate)


def apply_gaussian_filter_with_mask(type, array, mask, sd):
    truncate = np.ceil(2 * sd) / sd if sd != 0 else 0
    mask = np.expand_dims(mask, 0)
    mask = np.repeat(mask, array.shape[0], axis=0)
    if len(array.shape) == 4:
        if type == 'gauss_spatiotemporal':
            filtered = gaussian(array * mask, sigma=(0, sd, sd, sd), mode='nearest', truncate=truncate)
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered /= gaussian(mask, sigma=(0, sd, sd, sd), mode='nearest', truncate=truncate)
            filtered[np.logical_not(mask)] = 0
            return filtered
        elif type == 'gauss_spatial':
            filtered = gaussian(array * mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            intermed = gaussian(mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered /= gaussian(mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            filtered[np.logical_not(mask)] = 0
            return filtered
        else:
            raise NotImplementedError('Gaussian filter variant not implemented.')

def apply_gaussian_filter_with_mask_amc(type, array, mask, sd, spacing, sd_t = 2):
    [dx, dy, dz] = spacing
    # sd_z = (dx * sd) / dz
    truncate = np.ceil(2 * sd) / sd if sd != 0 else 0


    mask = np.expand_dims(mask, 0)
    mask = np.repeat(mask, array.shape[0], axis=0)
    if len(array.shape) == 4:
        if type == 'gauss_spatiotemporal':
            filtered = gaussian(array * mask, sigma=(sd_t, 0, sd, sd), mode='nearest', truncate=truncate)
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered /= gaussian(mask, sigma=(sd_t, 0, sd, sd), mode='nearest', truncate=truncate)
            filtered[np.logical_not(mask)] = 0
            return filtered
        elif type == 'gauss_spatial':
            filtered = gaussian(array * mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            # intermed = gaussian(mask, sigma=(0, sd, sd, sd), mode='nearest', truncate=truncate)
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered /= gaussian(mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            filtered[np.logical_not(mask)] = 0
            return filtered
        else:
            raise NotImplementedError('Gaussian filter variant not implemented.')

def apply_billateral_filter(array, mask, sigma_spatial):
    mask = np.expand_dims(mask, 1)
    mask = np.repeat(mask, array.shape[1], axis=1)
    filtered = np.zeros_like(array, dtype=np.float32)
    plt.imshow(array[4, 0], vmin=0, vmax=100)
    plt.show()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filtered[i, j] = denoise_bilateral(array[i, j], win_size=None,
                                               sigma_color=20, sigma_spatial=sigma_spatial,
                                               bins=10000)
    filtered *= mask
    plt.imshow(filtered[4, 0], vmin=0, vmax=100)
    plt.show()
    return filtered
def read_nii_folder(scans):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(scans)
    image = reader.Execute()
    data_dict = {'array': sitk.GetArrayFromImage(image).astype(np.float32),
                 'spacing': image.GetSpacing(),
                 'dims': image.GetSize(),
                 'shape': sitk.GetArrayFromImage(image).shape}
    return data_dict
def save_perfusion_parameters_amc(config, case, cbf_results, cbv_results, mtt_results, delay_results, tmax_results, data_dict):
    template = data_dict['cbf']
    if 'dwi_segmentation' in data_dict.keys():

        sitk.WriteImage(data_dict['dwi_segmentation'], os.path.join(wandb.run.dir, 'results', case, 'dwi_segmentation.nii.gz'))

    mask = np2itk(data_dict['mask'], template)
    sitk.WriteImage(mask, os.path.join(wandb.run.dir, 'results', case, 'vesselmask.nii.gz'))

    cbf_results = np2itk(cbf_results, template)
    sitk.WriteImage(cbf_results, os.path.join(wandb.run.dir, 'results', case, 'cbf.nii.gz'))

    cbv_results = np2itk(cbv_results, template)
    sitk.WriteImage(cbv_results, os.path.join(wandb.run.dir, 'results', case, 'cbv.nii.gz'))

    mtt_results = np2itk(mtt_results, template)
    sitk.WriteImage(mtt_results, os.path.join(wandb.run.dir, 'results', case, 'mtt.nii.gz'))

    delay_results = np2itk(delay_results, template)
    sitk.WriteImage(delay_results, os.path.join(wandb.run.dir, 'results', case, 'delay.nii.gz'))

    tmax_results = np2itk(tmax_results, template)
    sitk.WriteImage(tmax_results, os.path.join(wandb.run.dir, 'results', case, 'tmax.nii.gz'))

def save_perfusion_parameters(config, case, cbf_results, cbv_results, mtt_results, delay_results, tmax_results):
    mode = 'TRAINING' if config.mode == 'train' else 'TESTING'
    folder = r'data/ISLES2018'

    template_file = glob.glob(os.path.join(folder, mode, case, '*OT*', '*OT*.nii'))[0]
    template = sitk.ReadImage(template_file)
    sitk.WriteImage(template, os.path.join(wandb.run.dir, 'results', case, 'dwi_seg.nii'))

    cbf_results = np2itk(cbf_results, template)
    sitk.WriteImage(cbf_results, os.path.join(wandb.run.dir, 'results', case, 'cbf.nii'))

    cbv_results = np2itk(cbv_results, template)
    sitk.WriteImage(cbv_results, os.path.join(wandb.run.dir, 'results', case, 'cbv.nii'))

    mtt_results = np2itk(mtt_results, template)
    sitk.WriteImage(mtt_results, os.path.join(wandb.run.dir, 'results', case, 'mtt.nii'))

    delay_results = np2itk(delay_results, template)
    sitk.WriteImage(delay_results, os.path.join(wandb.run.dir, 'results', case, 'delay.nii'))

    tmax_results = np2itk(tmax_results, template)
    sitk.WriteImage(tmax_results, os.path.join(wandb.run.dir, 'results', case, 'tmax.nii'))

def np2itk(arr, original_img):
    img = sitk.GetImageFromArray(arr, False)
    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    img.CopyInformation(original_img)
    return img

# class CurveDataset(Dataset):
#     def __init__(self, data_curves, data_coordinates, collo_coordinates):
#         self.data_curves = data_curves
#         self.data_coordinates = data_coordinates
#         self.collo_coordinates = collo_coordinates
#     def __len__(self):
#         return len(self.data_curves)
#     def __getitem__(self, idx):
#         curves = torch.from_numpy(self.data_curves[idx]).float()
#         coordinates = torch.from_numpy(self.data_coordinates[idx]).float()
#         collocation = torch.from_numpy(self.collo_coordinates[idx]).float()
#
#         return curves, coordinates, collocation
# #
# class CurveDataset(Dataset):
#     def __init__(self, data_curves, data_coordinates, collo_coordinates, data_indices):
#         self.data_curves = data_curves
#         self.data_coordinates = data_coordinates
#         self.collo_coordinates = collo_coordinates
#         self.data_indices = data_indices
#
#     def __len__(self):
#         return len(self.data_curves)
#     def __getitem__(self, idx):
#         return self.data_curves[idx], self.data_coordinates[idx], self.collo_coordinates[idx], idx
class CurveDataset(Dataset):
    def __init__(self, data_curves, data_coordinates, collo_coordinates):
        self.data_curves = data_curves
        self.data_coordinates = data_coordinates
        self.collo_coordinates = collo_coordinates

    def __len__(self):
        return len(self.data_curves)
    def __getitem__(self, idx):
        return self.data_curves[idx], self.data_coordinates[idx], self.collo_coordinates[idx]