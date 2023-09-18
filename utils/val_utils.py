import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from einops.einops import repeat, rearrange
import pandas as pd
import torch
import torch.nn.functional as F
def load_nlr_results(cbv_ml=5, sd=2, undersample=False):
    if not undersample:

        result = sitk.ReadImage(rf'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\nlr_results\nlr_sd_{sd}.nii')
    else:
        result = sitk.ReadImage(rf'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\nlr_results\nlr_sd_{sd}_subsampled05.nii')

    result = sitk.GetArrayFromImage(result)
    result = result[:,cbv_ml-1,:,:]
    density = 1.05
    constant = (100 / density) * 0.55 / 0.75
    cbv = result[0]
    cbv = cbv * constant
    mtt = result[1]
    cbf = cbv / (mtt / 60)
    delay = result[2]
    tmax = delay + 0.5 * mtt

    return {'cbf': cbf,
            'mtt': mtt,
            'cbv': cbv,
            'delay': delay,
            'tmax': tmax}

def read_dcm_folder(folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image
def read_dcm(folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    series_dic = {series_id: reader.GetGDCMSeriesFileNames(folder, series_id) for series_id in
                  reader.GetGDCMSeriesIDs(folder)}
    series_names = ['mip', 'cbf', 'cbv', 'mtt', 'ttd', 'delay', 'tmax']
    data_dict = {}
    for name, id in zip(series_names, series_dic.keys()):
        reader = sitk.ImageSeriesReader()
        dicom_names = series_dic[id]
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        data_dict[name] = image
    return data_dict

def load_sygnovia_results(cbv_ml=5, sd=2, undersample=False):
    simulated_data_size = 32 * 7
    scan_center = 512 // 2
    simulated_data_start = scan_center - simulated_data_size // 2
    simulated_data_end = scan_center + simulated_data_size // 2
    if not undersample:
        data = read_dcm(rf'data/sygnovia_results/sd{sd}')
    else:
        data = read_dcm(rf'data/sygnovia_results/sd{sd}_undersample_05')
    # data['delay_cal'] = data['tmax'] - 0.5 * data['mtt']
    for key, val in data.items():
        array = sitk.GetArrayFromImage(val)[11:]
        array = array[cbv_ml-1]
        data[key] = array[simulated_data_start:simulated_data_end, simulated_data_start:simulated_data_end]
    return data

def load_sygnovia_results_amc(case):
    base_folder = "D:\PPINN_patient_data\AMCCTP\sygnovia"
    data_dict = {}
    data_dict['baseline'] = read_dcm_folder(rf"{base_folder}\{case}\Basic_Baseline")
    data_dict['avg'] = read_dcm_folder(rf"{base_folder}\{case}\Basic_Average")
    data_dict['cbf'] = read_dcm_folder(rf"{base_folder}\{case}\Results_CBFD")
    data_dict['cbv'] = read_dcm_folder(rf"{base_folder}\{case}\Results_CBVD")
    data_dict['mtt'] = read_dcm_folder(rf"{base_folder}\{case}\Results_MTTD")
    data_dict['tmax'] = read_dcm_folder(rf"{base_folder}\{case}\Results_TMAXD")
    data_dict['delay'] = data_dict['tmax'] - 0.5*data_dict['mtt']
    data_dict['core'] = read_dcm_folder(rf"{base_folder}\{case}\PenumbraCore_setting2\CoreBinaryMap")
    data_dict['penumbra'] = read_dcm_folder(rf"{base_folder}\{case}\PenumbraCore_setting2\PenumbraBinaryMap")
    os.makedirs(rf"{base_folder}\{case}\nifti", exist_ok=True)
    for key, val in data_dict.items():
        sitk.WriteImage(val, rf"{base_folder}\{case}\nifti\{key}.nii.gz")
    for key, val in data_dict.items():
        array = sitk.GetArrayFromImage(val)
        array[array<0] = 0
        # empty = np.zeros_like(array)
        # empty[array >= 0] = array[array >= 0]
        data_dict[key] = array

    return data_dict

def load_phantom_gt(cbv_ml=5, simulation_method=2, undersample='NA'):
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
    return perfusion_values_dict
def visualize_amc_sygno(case, slice, result_dict, data_dict):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = 150

    dwi_segmentation = sitk.GetArrayFromImage(data_dict['dwi_segmentation'])[slice]
    cbf_results = result_dict['cbf'][slice]
    cbv_results = result_dict['cbv'][slice]
    mtt_results = result_dict['mtt'][slice]
    delay_results = result_dict['delay'][slice]
    tmax_results = result_dict['tmax'][slice]
    sygno_segmentation = result_dict['core'][slice]
    penumbra_segmentation = result_dict['penumbra'][slice]

    if np.sum(cbf_results) == 0:
        return
    fig, ax = plt.subplots(1, 8, figsize = (10,5))

    ax[0].set_title('CBF', fontdict=font)
    im = ax[0].imshow(cbf_results, vmin=np.percentile(cbf_results[cbf_results>0],10), vmax=np.percentile(cbf_results[cbf_results>0],90), cmap='jet')
    cax = ax[0].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[1].set_title('MTT', fontdict=font)
    im = ax[1].imshow(mtt_results, vmin=np.percentile(mtt_results[mtt_results>0],10), vmax=np.percentile(mtt_results[mtt_results>0],90), cmap='jet')
    cax = ax[1].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[2].set_title('CBV', fontdict=font)
    im = ax[2].imshow(cbv_results, vmin=np.percentile(cbv_results[cbv_results>0],10), vmax=np.percentile(cbv_results[cbv_results>0],90), cmap='jet')
    cax = ax[2].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[3].set_title('Delay', fontdict=font)
    im = ax[3].imshow(delay_results, vmin=np.percentile(delay_results[delay_results>0],10), vmax=np.percentile(delay_results[delay_results>0],90), cmap='jet')
    cax = ax[3].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('s', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[4].set_title('Tmax', fontdict=font)
    im = ax[4].imshow(tmax_results, vmin=np.percentile(tmax_results[tmax_results>0],10), vmax=np.percentile(tmax_results[tmax_results>0],90), cmap='jet')
    cax = ax[4].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('s', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[5].set_title('S. pen.', fontdict=font)
    im = ax[5].imshow(penumbra_segmentation, vmin=0, vmax=2, cmap='jet')

    ax[6].set_title('S. core', fontdict=font)
    im = ax[6].imshow(sygno_segmentation, vmin=0, vmax=2, cmap='jet')

    ax[7].set_title('DWI seg', fontdict=font)
    im = ax[7].imshow(dwi_segmentation, vmin=0, vmax=2, cmap='jet')
    # cax = ax[5].inset_axes([0, -0.2, 1, 0.1])
    # bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    # bar.outline.set_color('black')
    # bar.set_label('s', fontdict=font)
    # bar.ax.tick_params(labelsize=14)

    for i in range(7):
        ax[i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    wandb.log({"results_sygno_{}".format(case): plt})
    # plt.show()
    plt.close()
def visualize_amc(case, slice, result_dict, data_dict):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = 150

    dwi_segmentation = sitk.GetArrayFromImage(data_dict['dwi_segmentation'])[slice]
    cbf_results = result_dict['cbf']
    cbv_results = result_dict['cbv']
    mtt_results = result_dict['mtt']
    delay_results = result_dict['delay']
    tmax_results = result_dict['tmax']

    fig, ax = plt.subplots(1, 6, figsize = (10,5))

    ax[0].set_title('CBF', fontdict=font)
    im = ax[0].imshow(cbf_results, vmin=np.percentile(cbf_results[cbf_results>0],10), vmax=np.percentile(cbf_results[cbf_results>0],90), cmap='jet')
    cax = ax[0].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[1].set_title('MTT', fontdict=font)
    im = ax[1].imshow(mtt_results, vmin=np.percentile(mtt_results[mtt_results>0],10), vmax=np.percentile(mtt_results[mtt_results>0],90), cmap='jet')
    cax = ax[1].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[2].set_title('CBV', fontdict=font)
    im = ax[2].imshow(cbv_results, vmin=np.percentile(cbv_results[cbv_results>0],10), vmax=np.percentile(cbv_results[cbv_results>0],90), cmap='jet')
    cax = ax[2].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[3].set_title('Delay', fontdict=font)
    im = ax[3].imshow(delay_results, vmin=np.percentile(delay_results[delay_results>0],10), vmax=np.percentile(delay_results[delay_results>0],90), cmap='jet')
    cax = ax[3].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('s', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[4].set_title('Tmax', fontdict=font)
    im = ax[4].imshow(tmax_results, vmin=np.percentile(tmax_results[tmax_results>0],10), vmax=np.percentile(tmax_results[tmax_results>0],90), cmap='jet')
    cax = ax[4].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('s', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[5].set_title('DWI seg', fontdict=font)
    im = ax[5].imshow(dwi_segmentation, vmin=0, vmax=2, cmap='jet')
    # cax = ax[5].inset_axes([0, -0.2, 1, 0.1])
    # bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    # bar.outline.set_color('black')
    # bar.set_label('s', fontdict=font)
    # bar.ax.tick_params(labelsize=14)

    for i in range(6):
        ax[i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    # wandb.log({"results_{}".format(case): plt})
    # plt.show()
    plt.close()
def visualize(slice, case, perfusion_values, result_dict):

    cbf_results = result_dict['cbf'].cpu().detach().numpy()
    cbv_results = result_dict['cbv'].cpu().detach().numpy()
    mtt_results = result_dict['mtt'].cpu().detach().numpy()
    delay_results = result_dict['delay'].cpu().detach().numpy()
    tmax_results = result_dict['tmax'].cpu().detach().numpy()

    isles_cbf = perfusion_values[..., 0]
    isles_cbv = perfusion_values[..., 1]
    isles_mtt = perfusion_values[..., 2]
    isles_tmax = perfusion_values[..., 3]
    isles_gt_core = perfusion_values[..., 4]
    isles_delay = perfusion_values[..., 5]

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    if np.sum(cbf_results) == 0:
        return

    if np.sum(cbv_results) == 0:
        return
    if np.sum(isles_cbv) == 0:
        return
    fig, ax = plt.subplots(2, 6, figsize=(12, 6))

    ax[0, 0].set_title('CBF', fontdict=font)

    im = ax[0, 0].imshow(cbf_results, vmin=np.percentile(cbf_results[cbf_results>0],10), vmax=np.percentile(cbf_results[cbf_results>0],90), cmap='jet')
    cax = ax[0, 0].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[1, 0].imshow(isles_cbf, vmin=np.percentile(isles_cbf[isles_cbf>0], 10), vmax=np.percentile(isles_cbf[isles_cbf>0],90), cmap='jet')
    cax = ax[1, 0].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 1].set_title('MTT', fontdict=font)

    im = ax[0, 1].imshow(mtt_results, vmin=np.percentile(mtt_results[mtt_results>0],10), vmax=np.percentile(mtt_results[mtt_results>0],90), cmap='jet')
    cax = ax[0, 1].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[1, 1].imshow(isles_mtt, vmin=np.percentile(isles_mtt[isles_mtt>0], 10), vmax=np.percentile(isles_mtt[isles_mtt>0],90), cmap='jet')
    cax = ax[1, 1].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 2].set_title('CBV', fontdict=font)

    im = ax[0, 2].imshow(cbv_results, vmin=np.percentile(cbv_results[cbv_results>0],10), vmax=np.percentile(cbv_results[cbv_results>0],90), cmap='jet')
    cax = ax[0, 2].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[1, 2].imshow(isles_cbv, vmin=np.percentile(isles_cbv[isles_cbv>0], 10), vmax=np.percentile(isles_cbv[isles_cbv>0],90), cmap='jet')
    cax = ax[1, 2].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 3].set_title('Tmax', fontdict=font)
    im = ax[0, 3].imshow(tmax_results, vmin=np.percentile(tmax_results[tmax_results>0],10), vmax=np.percentile(tmax_results[tmax_results>0],90), cmap='jet')
    cax = ax[0, 3].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    im = ax[1, 3].imshow(isles_tmax, vmin=np.percentile(isles_tmax[isles_tmax>0], 10), vmax=np.percentile(isles_tmax[isles_tmax>0],90), cmap='jet')
    cax = ax[1, 3].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 4].set_title('Delay', fontdict=font)
    im = ax[0, 4].imshow(delay_results, vmin=np.percentile(delay_results[delay_results>0],10), vmax=np.percentile(delay_results[delay_results>0],90), cmap='jet')
    cax = ax[0, 4].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    im = ax[1, 4].imshow(isles_delay, vmin=np.percentile(isles_delay[isles_delay>0], 10), vmax=np.percentile(isles_delay[isles_delay>0],90), cmap='jet')
    cax = ax[1, 4].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 5].set_title('Core', fontdict=font)
    ax[1, 5].set_title('Core', fontdict=font)

    im = ax[0, 5].imshow(isles_gt_core, vmin=0.01, vmax=2, cmap='jet')
    cax = ax[0, 5].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    im = ax[1, 5].imshow(isles_gt_core, vmin=0.01, vmax=2, cmap='jet')
    cax = ax[1, 5].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    # for i in range(5):
    #     ax[1, i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    ax[0, 0].set_ylabel('PPINN', fontdict=font)
    ax[1, 0].set_ylabel('ISLES/Rapid', fontdict=font)
    # plt.tight_layout()
    wandb.log({"results_{}".format(case): plt})

    plt.close()


def plot_software_results(results_dict, phantom_dict, name='Sygno.via'):
    cbf = results_dict['cbf']
    cbv = results_dict['cbv']
    mtt = results_dict['mtt']
    tmax = results_dict['tmax']
    delay = results_dict['delay']

    gt_cbf = phantom_dict['cbf']
    gt_cbv = phantom_dict['cbv']
    gt_mtt = phantom_dict['mtt']
    gt_delay = phantom_dict['delay']
    gt_tmax = phantom_dict['tmax']


    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    fig, ax = plt.subplots(2, 5, figsize=(10,6))
    ax[0, 0].set_title('CBF', fontdict=font)
    ax[0, 0].imshow(cbf, vmin=0, vmax=100, cmap='jet')
    im = ax[1, 0].imshow(gt_cbf, vmin=0, vmax=100, cmap='jet')
    cax = ax[2, 0].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    ax[0, 0].set_ylabel(f'{name}', fontdict=font)
    ax[1, 0].set_ylabel('GT', fontdict=font)

    ax[0, 1].set_title('MTT', fontdict=font)
    ax[0, 1].imshow(mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    im = ax[1, 1].imshow(gt_mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    cax = ax[2, 1].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 2].set_title('CBV', fontdict=font)
    ax[0, 2].imshow(cbv, vmin=0.01, vmax=7, cmap='jet')
    im = ax[1, 2].imshow(gt_cbv, vmin=0.01, vmax=7, cmap='jet')
    cax = ax[2, 2].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 3].set_title('Delay', fontdict=font)
    ax[0, 3].imshow(delay, vmin=0.01, vmax=3.5, cmap='jet')
    im = ax[1, 3].imshow(gt_delay, vmin=0.01, vmax=3.5, cmap='jet')
    cax = ax[2, 3].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 4].set_title('Tmax', fontdict=font)
    ax[0, 4].imshow(tmax, vmin=0.01, vmax=15, cmap='jet')
    im = ax[1, 4].imshow(gt_tmax, vmin=0.01, vmax=15, cmap='jet')
    cax = ax[2, 4].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    for i in range(5):
        ax[2, i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    fig.suptitle(f'{name} vs. Phantom ground truth', fontdict=font)
    plt.tight_layout()
    os.makedirs(os.path.join(wandb.run.dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(wandb.run.dir, 'plots', f'{name}_vs_gt.png'), dpi=150)
    # plt.show()

def plot_software_results_on_axis(ax, results_dict, name='Sygno.via', title=False):

    cbf = results_dict['cbf']
    cbv = results_dict['cbv']
    mtt = results_dict['mtt']
    tmax = results_dict['tmax']
    delay = results_dict['delay']

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    # fig, ax = plt.subplots(3, 5, figsize=(10,6))
    if title:
        ax[0].set_title('CBF', fontdict=font)
        ax[1].set_title('MTT', fontdict=font)
        ax[2].set_title('CBV', fontdict=font)
        ax[3].set_title('Delay', fontdict=font)
        ax[4].set_title('Tmax', fontdict=font)

    ax[0].imshow(cbf, vmin=0, vmax=100, cmap='jet')
    ax[0].set_ylabel(f'{name}', fontdict=font)
    ax[1].imshow(mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    ax[2].imshow(cbv, vmin=0.01, vmax=7, cmap='jet')
    ax[3].imshow(delay, vmin=0.01, vmax=3.5, cmap='jet')
    ax[4].imshow(tmax, vmin=0.01, vmax=15, cmap='jet')

    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
def plot_software_difference_on_axis(ax, results_dict, gt_dict, name='Sygno.via', title=False, error=None):

    if error == 'nmse':
        cbf = ((results_dict['cbf'] - gt_dict['cbf'])/gt_dict['cbf'])**2
        cbv = ((results_dict['cbv'] - gt_dict['cbv'])/gt_dict['cbv'])**2
        mtt = ((results_dict['mtt'] - gt_dict['mtt'])/gt_dict['mtt'])**2
        tmax = ((results_dict['tmax'] - gt_dict['tmax'])/gt_dict['tmax'])**2
        delay = ((results_dict['delay'] - gt_dict['delay'])/gt_dict['delay'])**2

    elif error == 'nmae':
        cbf = np.abs(((results_dict['cbf'] - gt_dict['cbf'])/gt_dict['cbf']))
        cbv = np.abs(((results_dict['cbv'] - gt_dict['cbv'])/gt_dict['cbv']))
        mtt = np.abs(((results_dict['mtt'] - gt_dict['mtt'])/gt_dict['mtt']))
        tmax = np.abs(((results_dict['tmax'] - gt_dict['tmax'])/gt_dict['tmax']))
        delay = np.abs(((results_dict['delay'] - gt_dict['delay'])/gt_dict['delay']))
    else:
        cbf = results_dict['cbf'] - gt_dict['cbf']
        cbv = results_dict['cbv'] - gt_dict['cbv']
        mtt = results_dict['mtt'] - gt_dict['mtt']
        tmax = results_dict['tmax'] - gt_dict['tmax']
        delay = results_dict['delay'] - gt_dict['delay']

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    # fig, ax = plt.subplots(3, 5, figsize=(10,6))
    if title:
        ax[0].set_title('CBF', fontdict=font)
        ax[1].set_title('MTT', fontdict=font)
        ax[2].set_title('CBV', fontdict=font)
        ax[3].set_title('Delay', fontdict=font)
        ax[4].set_title('Tmax', fontdict=font)

    ax[0].set_ylabel(f'{name}', fontdict=font)
    if not error:
        ax[0].imshow(cbf, vmin=-10, vmax=10, cmap='jet')
        ax[1].imshow(mtt, vmin=-2, vmax=2, cmap='jet')
        ax[2].imshow(cbv, vmin=-1, vmax=1, cmap='jet')
        ax[3].imshow(delay, vmin=-1, vmax=1, cmap='jet')
        ax[4].imshow(tmax, vmin=-1, vmax=1, cmap='jet')
    else:
        ax[0].imshow(cbf,vmin=-1,vmax=1, cmap='jet')
        ax[1].imshow(mtt,vmin=-1,vmax=1, cmap='jet')
        ax[2].imshow(cbv,vmin=-1,vmax=1, cmap='jet')
        ax[3].imshow(delay,vmin=-1,vmax=1, cmap='jet')
        ax[4].imshow(tmax,vmin=-1,vmax=1, cmap='jet')
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
def add_colorbars_to_fig(phantom_dict, fig, ax, gt_axis):
    cbar_axis = gt_axis + 1
    gt_cbf = phantom_dict['cbf']
    gt_cbv = phantom_dict['cbv']
    gt_mtt = phantom_dict['mtt']
    gt_delay = phantom_dict['delay']
    gt_tmax = phantom_dict['tmax']

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150
    # [x0, y0, width, height]
    im = ax[gt_axis, 0].imshow(gt_cbf, vmin=0, vmax=100, cmap='jet')
    cax = ax[cbar_axis, 0].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    ax[gt_axis, 0].set_ylabel('GT', fontdict=font)
    ax[gt_axis, 1].set_ylabel(' ', fontdict=font)

    im = ax[gt_axis, 1].imshow(gt_mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    cax = ax[cbar_axis, 1].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 2].imshow(gt_cbv, vmin=0.01, vmax=7, cmap='jet')
    cax = ax[cbar_axis, 2].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 3].imshow(gt_delay, vmin=0.01, vmax=3.5, cmap='jet')
    cax = ax[cbar_axis, 3].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 4].imshow(gt_tmax, vmin=0.01, vmax=15, cmap='jet')
    cax = ax[cbar_axis, 4].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    for i in range(5):
        ax[cbar_axis, i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    # plt.tight_layout()

def log_software_results(results, cbv_ml, corrected=False, wandb_on=True):
    for metric in ['mse', 'mae', 'me']:
        table = []
        for key in ['cbf', 'cbv', 'mtt', 'delay', 'tmax']:
            if metric == 'mse':
                table.append([cbv_ml, f"{key}",
                              # np.mean((results['nlr'][key] - results['gt'][key]) ** 2),
                              # np.mean((results['sygnovia'][key] - results['gt'][key]) ** 2),
                              np.mean((results['sppinn'][key] - results['gt'][key]) ** 2)])
            elif metric == 'mae':
                table.append([cbv_ml, f"{key}",
                              # np.mean(np.abs(results['nlr'][key] - results['gt'][key])),
                              # np.mean(np.abs(results['sygnovia'][key] - results['gt'][key])),
                              np.mean(np.abs(results['sppinn'][key] - results['gt'][key]))])
            elif metric == 'me':
                table.append([cbv_ml, f"{key}",
                              # np.mean(results['nlr'][key] - results['gt'][key]),
                              # np.mean(results['sygnovia'][key] - results['gt'][key]),
                              np.mean(results['sppinn'][key] - results['gt'][key])])
            else:
                raise NotImplementedError('Not implemented')
        columns = ['cbv_ml', 'parameter','SPPINN']
        df = pd.DataFrame(columns=['cbv_ml', 'parameter', 'SPPINN'], data=table)
        if wandb_on:
            wandb_table = wandb.Table(data=df)
            # wandb_table = wandb.Table(data=table, columns=columns)
            wandb.log({f'table_{metric}': wandb_table}) if not corrected else wandb.log({f'table_{metric}_cor': wandb_table})
        else:
            return df
def drop_edges(results):
    skip_rows = sorted(list(range(-2, 226, 32)) + list(range(-1, 226, 32)) + list(range(0, 226, 32)) + list(range(1, 226, 32)))
    skip_rows = skip_rows[2:-2]
    for k1 in results.keys():
        for k2 in ['cbf', 'cbv', 'mtt', 'delay', 'tmax']:
            results[k1][k2] = np.delete(results[k1][k2], skip_rows, axis=0)
            results[k1][k2] = np.delete(results[k1][k2], skip_rows, axis=1)
    return results
def drop_edges_per_method(results):
    skip_rows = sorted(list(range(-2, 226, 32)) + list(range(-1, 226, 32)) + list(range(0, 226, 32)) + list(range(1, 226, 32)))
    skip_rows = skip_rows[2:-2]
    for k2 in ['cbf', 'cbv', 'mtt', 'delay', 'tmax']:
        results[k2] = np.delete(results[k2], skip_rows, axis=0)
        results[k2] = np.delete(results[k2], skip_rows, axis=1)
    return results
def drop_unphysical(results):
    for k1 in results.keys():
        results[k1]['cbf'] = np.clip(results[k1]['cbf'], a_min=0 , a_max=150)
        results[k1]['cbv'] = np.clip(results[k1]['cbv'], a_min=0, a_max=10)
        results[k1]['mtt'] = np.clip(results[k1]['mtt'], a_min=0 , a_max=30)
        results[k1]['delay'] = np.clip(results[k1]['delay'], a_min=0 , a_max=10)
        results[k1]['tmax'] = np.clip(results[k1]['tmax'], a_min=0 , a_max=20)
    return results
def drop_unphysical_per_method(results):
    results['cbf'] = np.clip(results['cbf'], a_min=0 , a_max=150)
    results['cbv'] = np.clip(results['cbv'], a_min=0, a_max=10)
    results['mtt'] = np.clip(results['mtt'], a_min=0 , a_max=30)
    results['delay'] = np.clip(results['delay'], a_min=0 , a_max=10)
    results['tmax'] = np.clip(results['tmax'], a_min=0 , a_max=20)
    return results
def drop_unphysical_amc(results):
    results['cbf'] = np.clip(results['cbf'], a_min=0 , a_max=150)
    results['cbv'] = np.clip(results['cbv'], a_min=0, a_max=10)
    results['mtt'] = np.clip(results['mtt'], a_min=0 , a_max=30)
    results['delay'] = np.clip(results['delay'], a_min=0 , a_max=10)
    results['tmax'] = np.clip(results['tmax'], a_min=0 , a_max=20)
    return results

def drop_unphysical_amc2(results):
    results['cbf'] = np.clip(results['cbf'], a_min=0 , a_max=1000)
    results['cbv'] = np.clip(results['cbv'], a_min=0, a_max=20)
    results['mtt'] = np.clip(results['mtt'], a_min=0 , a_max=50)
    results['delay'] = np.clip(results['delay'], a_min=0 , a_max=20)
    results['tmax'] = np.clip(results['tmax'], a_min=0 , a_max=50)
    return results

def plot_results(results, corrected=False):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    fig, ax = plt.subplots(3, 5, figsize=(14, 14 / 5 * 3))
    plot_software_results_on_axis(ax[0], results['sppinn'], name='SPPINN')
    plot_software_results_on_axis(ax[1], results['gt'], name='GT')

    # ax = color_axes(ax)
    add_colorbars_to_fig(results['gt'], fig, ax, 1)
    # plt.tight_layout()
    os.makedirs(os.path.join(wandb.run.dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(wandb.run.dir, 'plots', f'software_vs_gt.png'), dpi=150)
    wandb.log({"results_compare": plt}) if not corrected else wandb.log({"results_compare_cor": plt})
def color_axes(ax):
    ax[1,0].spines['bottom'].set_color('red')
    ax[1,0].spines['top'].set_color('red')
    ax[1,0].spines['left'].set_color('red')
    ax[1,0].spines['right'].set_color('red')
    ax[1,4].spines['bottom'].set_color('red')
    ax[1,4].spines['top'].set_color('red')
    ax[1,4].spines['left'].set_color('red')
    ax[1,4].spines['right'].set_color('red')

    ax[2,2].spines['bottom'].set_color('red')
    ax[2,2].spines['top'].set_color('red')
    ax[2,2].spines['left'].set_color('red')
    ax[2,2].spines['right'].set_color('red')
    ax[2,4].spines['bottom'].set_color('red')
    ax[2,4].spines['top'].set_color('red')
    ax[2,4].spines['left'].set_color('red')
    ax[2,4].spines['right'].set_color('red')
    return ax