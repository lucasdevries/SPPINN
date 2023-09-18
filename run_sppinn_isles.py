import torch
import torch.nn as nn
import numpy as np
from models.MLP_st import MLP, MLP_ODE, MLP_siren, MLP_ODE_siren, MLP_sirenlike
from utils.train_utils import AverageMeter, set_seed
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os, glob
import wandb
from utils.data_utils import load_ISLES_data, save_perfusion_parameters_amc
from utils.val_utils import drop_unphysical_amc2
import re
from torchsummary import summary
from utils.data_utils import CurveDataset
import pickle
# from utils.val_utils import load_nlr_results, plot_curves_at_epoch_amc_st
from utils.val_utils import load_nlr_results, visualize_amc
from utils.val_utils import log_software_results, plot_results, drop_edges, drop_unphysical, drop_unphysical_amc, visualize_amc_sygno
from siren_pytorch import SirenNet
from einops.einops import rearrange, repeat
import argparse
class SPPINN(nn.Module):
    def __init__(self,
                 config,
                 shape_in,
                 # perfusion_values,
                 n_inputs=1,
                 std_t=1,
                 original_data_shape = None,
                 original_indices = None,
                 case='C102',
                 slice=10,
                 first=False):
        super(SPPINN, self).__init__()
        self.config = config
        self.case = case
        self.slice = slice
        print(f"Processing case {self.case}, slice {self.slice}.")

        self.PID = os.getpid()
        self.logger = logging.getLogger(str(self.PID))
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda:{}".format(self.config.gpu_device))
            torch.cuda.set_device("cuda:{}".format(self.config.gpu_device))
            self.logger.info("Operation will be on *****GPU-CUDA{}***** ".format(self.config.gpu_device))
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.lw_data, self.lw_res, self.lw_bc = (0, 0, 0)
        self.optimizer = None
        self.scheduler = None
        # self.max_epochs = self.config.epochs
        self.first = first
        if first:
            self.max_epochs = 2*self.config.epochs
        else:
            self.max_epochs = self.config.epochs
        self.milestones = [self.max_epochs//3, 2*self.max_epochs//3]
        self.gamma = self.config.gamma
        # self.milestones = [3*self.config.epochs]

        self.interpolator = None
        self.var_list = None
        self.shape_in = shape_in
        self.original_data_shape = original_data_shape
        self.original_data_indices = original_indices
        self.std_t = std_t
        self.neurons_out = 1
        # self.perfusion_values = perfusion_values
        # initialize flow parameters
        self.log_domain = config.log_domain
        self.flow_cbf = None
        self.flow_t_delay = None
        self.flow_mtt = None
        self.aif_training_off = False
        self.tissue_training_off = False
        n_layers = config.n_layers
        n_units = config.n_units
        self.lr = config.lr
        loss_weights = (config.lw_data, config.lw_res, 0)
        bn = config.bn
        self.batch_size = config.batch_size
        self.data_coordinates_xy = None

        if self.config.siren:
            print('using sirens')
            self.NN_tissue = MLP_siren(
                                    dim_in=3,  # input dimension, ex. 2d coor
                                    dim_hidden=self.config.hidden_tissue,  # hidden dimension
                                    dim_out=1,  # output dimension, ex. rgb value
                                    num_layers=3,  # number of layers
                                    final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                                    w0_initial=self.config.siren_w0
                                    # different signals may require different omega_0 in the first layer - this is a hyperparameter
                                )
            self.NN_ode = MLP_ODE_siren(
                                            dim_in=2,  # input dimension, ex. 2d coor
                                            dim_hidden=self.config.hidden_ode,  # hidden dimension normal 16
                                            dim_out=3,  # output dimension, ex. rgb value
                                            num_layers=3,  # number of layers
                                            final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                                            w0_initial=self.config.siren_w0
                                            # different signals may require different omega_0 in the first layer - this is a hyperparameter
                                        )

        else:
            print('using MLPs')
            self.NN_tissue = MLP(
                False,
                n_layers,
                self.config.hidden_tissue,
                n_inputs=3,
                neurons_out=1,
                bn=bn,
                act='tanh'
            )

            self.NN_ode = MLP_ODE(
                n_layers,
                self.config.hidden_ode,
                n_inputs=2,
                neurons_out=3,
                bn=bn,
                act='tanh'
            )


        self.NN_aif = MLP(
            True,
            n_layers,
            n_units,
            n_inputs=n_inputs,
            neurons_out=1,
            bn=bn,
            act='tanh'
        )

        self.current_iteration = 0
        self.epoch = 1

        self.init_from_previous_slice = self.config.init_from_previous_slice
        self.init_model_parameters()
        self.set_lr(self.config.optimizer, self.lr)

        self.set_loss_weights(loss_weights)
        self.set_params_to_domain()
        self.set_device(self.device)
        self.float()
    # @profile
    def forward_NNs(self, aif_time, txy):
        t = txy[...,:1]
        xy = txy[...,1:]

        c_tissue = self.NN_tissue(t, xy)
        c_aif = self.NN_aif(aif_time, xy)
        return c_aif, c_tissue

    # @profile
    def forward_complete(self, aif_time, txy):
        t = txy[...,:1]
        xy = txy[...,1:]
        # t = t.unsqueeze(-1)        steps = t.shape[0]
        # Get NN output: a tissue curve for each voxel
        c_tissue = self.NN_tissue(t, xy)
        c_aif = self.NN_aif(aif_time, xy)
        # Get time-derivative of tissue curve
        c_tissue_dt = (1 / self.std_t) * self.__fwd_gradients(c_tissue, t)

        t = t.detach()
        t.requires_grad = False

        params = self.NN_ode(xy)
        if self.log_domain:
            cbf = torch.exp(params[..., :1])
            mtt = 24*torch.exp(params[..., 1:2])
            delay = 3*torch.exp(params[..., 2:])
        else:
            cbf = params[..., :1]
            mtt = 24*params[..., 1:2]
            delay = 3*params[..., 2:]

        c_aif_a = self.NN_aif(t - delay / self.std_t, xy) # 128
        c_aif_b = self.NN_aif(t - delay / self.std_t - mtt / self.std_t, xy)

        residual = c_tissue_dt - cbf * (c_aif_a - c_aif_b).unsqueeze(-1)
        return c_aif, c_tissue, residual

    def set_loss_weights(self, loss_weights):
        loss_weights = torch.tensor(loss_weights)
        self.lw_data, self.lw_res, self.lw_bc = loss_weights
        self.lw_data.to(self.device)
        self.lw_res.to(self.device)
        self.lw_bc.to(self.device)

    def set_lr(self, optimizer, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr) if optimizer == 'Adam' else torch.optim.SGD(
            self.parameters(), lr=lr)
        if self.config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=self.milestones,
                                                              gamma=self.gamma)

    def init_model_parameters(self):
        if self.init_from_previous_slice:
            print('Initializing parameters from previous slice if available')
            base = os.path.join(wandb.run.dir,'models',f'{self.case}')
            os.makedirs(base, exist_ok=True)
            if os.path.isdir(base):
                if len(os.listdir(base)) == 3:
                    self.NN_aif.load_state_dict(torch.load(os.path.join(base, f'aif_case_{self.case}_first.pth')))
                    self.NN_tissue.load_state_dict(torch.load(os.path.join(base, f'tac_case_{self.case}_first.pth')))
                    self.NN_ode.load_state_dict(torch.load(os.path.join(base, f'ode_case_{self.case}_first.pth')))
                    print(f"Models intitialized with slice first.")
                elif len(os.listdir(base)) > 3:
                    available_slices = [int(x.split("_")[-1].split('.')[0][2:]) for x in glob.glob(os.path.join(base, f'aif_case_{self.case}_sl*.pth'))]
                    #find closest
                    closest_slice = min(available_slices, key=lambda x: abs(x - self.slice))
                    self.NN_aif.load_state_dict(torch.load(os.path.join(base, f'aif_case_{self.case}_sl{closest_slice}.pth')))
                    self.NN_tissue.load_state_dict(torch.load(os.path.join(base, f'tac_case_{self.case}_sl{closest_slice}.pth')))
                    self.NN_ode.load_state_dict(torch.load(os.path.join(base, f'ode_case_{self.case}_sl{closest_slice}.pth')))
                    print(f"Models intitialized with slice {closest_slice}.")
                else:
                    print("No earlier slices available.")
            else:
                print("Given directory doesn't exist")

    def set_device(self, device):
        self.to(device)

    def set_params_to_domain(self):
        for name, param in self.named_parameters():
            if 'flow' in name:
                param.data = torch.log(param.data) if self.log_domain else param.data

    def get_ode_params(self):

        params = self.NN_ode(self.data_coordinates_xy)

        result_cbf = torch.zeros([256, 256])
        result_mtt = torch.zeros([256, 256])
        result_delay = torch.zeros([256, 256])

        result_cbf[self.original_data_indices] = params[...,0].cpu()
        result_mtt[self.original_data_indices] = params[...,1].cpu()
        result_delay[self.original_data_indices] = params[...,2].cpu()

        self.flow_cbf = result_cbf.unsqueeze(-1)
        self.flow_mtt = result_mtt.unsqueeze(-1)
        self.flow_t_delay = result_delay.unsqueeze(-1)

        if self.log_domain:
            return [torch.exp(self.flow_cbf), 24 * torch.exp(self.flow_mtt), 3 * torch.exp(self.flow_t_delay)]
        else:
            return [self.flow_cbf, 24 * self.flow_mtt, 3 * self.flow_t_delay]


    def get_delay(self, seconds=True):
        _, _, delay = self.get_ode_params()
        if seconds:
            return delay
        else:
            return delay / 60

    def get_mtt(self, seconds=True):
        _, mtt, _ = self.get_ode_params()
        if seconds:
            return mtt.squeeze(-1)
        else:
            return mtt.squeeze(-1) / 60

    def get_cbf(self, seconds=True):
        density = 1.05
        constant = (100 / density) * 0.55 / 0.75
        constant = torch.as_tensor(constant)#.to(self.device)
        flow, _, _ = self.get_ode_params()
        if seconds:
            return constant * flow
        else:
            return constant * flow * 60

    def fit(self,
            slice,
            data_dict,
            batch_size,
            epochs,
            case):

        data_time = data_dict['time'][slice]
        data_aif = data_dict['aif']
        data_aif_time = data_dict['aif_time'].to(self.device)
        data_curves = data_dict['curves'][slice].numpy()
        data_curves = data_curves[self.original_data_indices]
        data_coordinates = data_dict['coordinates'][slice].numpy()
        data_coordinates = data_coordinates[self.original_data_indices]
        timepoints = len(data_dict['aif'])

        self.data_coordinates_xy = data_dict['coordinates_xy_only'][self.original_data_indices].to(self.device)
        # self.data_coordinates_xy = data_dict['coordinates_xy_only']

        data_boundary = data_dict['bound']


        collocation_txys = np.zeros(data_coordinates.shape)
        collocation_txys[..., 1:] = data_coordinates[..., 1:]
        # collocation_txys = data_coordinates.clone()
        data_curves = rearrange(data_curves, 'dum1 (t val)-> (dum1  t) val', val=1)
        data_coordinates = rearrange(data_coordinates, 'dum1 t val-> (dum1 t) val')
        collocation_coordinates = rearrange(collocation_txys, 'dum1 t val -> (dum1 t) val')

        for ep in tqdm(range(self.current_iteration + 1, self.current_iteration + self.max_epochs + 1)):
            collocation_coordinates[:, 0] = torch.FloatTensor(*collocation_coordinates.shape[:-1]).uniform_(
                 torch.min(data_time), torch.max(data_time)
            )
            epoch_aif_loss = AverageMeter()
            epoch_tissue_loss = AverageMeter()
            epoch_residual_loss = AverageMeter()
            iter = 0

            integers = np.arange(len(data_curves))
            np.random.shuffle(integers)
            splits = np.array_split(integers, int(len(data_curves)/batch_size))

            for split in splits:

                batch_curves = torch.from_numpy(data_curves[split]).float().to(self.device)
                batch_coordinates = torch.from_numpy(data_coordinates[split]).float().to(self.device)
                batch_collo = torch.from_numpy(collocation_coordinates[split]).float().to(self.device)

                batch_aif = data_aif.to(self.device)
                batch_time = data_time[:, 0, 0, :].to(self.device)
                batch_aif_time = data_aif_time[:, 0, 0, :].to(self.device)
                batch_boundary = data_boundary.to(self.device)

                loss_aif, loss_tissue, loss_residual = self.optimize(batch_time,
                                                                     batch_coordinates,
                                                                     batch_aif,
                                                                     batch_curves,
                                                                     batch_boundary,
                                                                     batch_collo,
                                                                     batch_aif_time
                                                                     )
                epoch_aif_loss.update(loss_aif.item())
                epoch_tissue_loss.update(loss_tissue.item())
                epoch_residual_loss.update(loss_residual.item())
                self.min_aif_loss = loss_aif.item()

                iter+=1

            # self.forward_complete_check(batch_time, derivative_coordinates)
            if self.config.use_scheduler:
                self.scheduler.step()

            # data_curves = data_curves.cpu().detach().numpy()
            if self.config.wandb:
                metrics = {f"aif_loss_{case}_{slice}": epoch_aif_loss.avg,
                           f"tissue_loss_{case}_{slice}": epoch_tissue_loss.avg,
                           f"residual_loss_{case}_{slice}": epoch_residual_loss.avg,
                           f"lr_{case}_{slice}": self.optimizer.param_groups[0]['lr'],
                           }
                wandb.log(metrics)

            # plot intermediate results
            if ep % 25 == 0:
                cbf = self.get_cbf(seconds=False).squeeze()
                mtt = self.get_mtt(seconds=True).squeeze()
                mtt_min = self.get_mtt(seconds=False).squeeze()
                delay = self.get_delay(seconds=True).squeeze()
                cbv = cbf * mtt_min
                tmax = delay + 0.5 * mtt

                result_dict = {'cbf': cbf,
                               'cbv': cbv,
                               'mtt': mtt,
                               'delay': delay,
                               'tmax': tmax}
                # result_dict = drop_unphysical_amc(result_dict)
                mask_data = data_dict['mask'][slice]
                mask_data = mask_data.cpu().numpy()
                for key in result_dict.keys():
                    result_dict[key] = result_dict[key].cpu().detach().numpy()
                    result_dict[key] *= mask_data
                visualize_amc(case, slice, result_dict, data_dict)

            self.current_iteration += 1
            self.epoch += 1


        # save models

        if self.init_from_previous_slice and not self.first:
            torch.save(self.NN_aif.state_dict(), os.path.join(wandb.run.dir, 'models',f'{case}', f'aif_case_{case}_sl{slice}.pth'))
            torch.save(self.NN_tissue.state_dict(), os.path.join(wandb.run.dir, 'models',f'{case}', f'tac_case_{case}_sl{slice}.pth'))
            torch.save(self.NN_ode.state_dict(), os.path.join(wandb.run.dir, 'models',f'{case}', f'ode_case_{case}_sl{slice}.pth'))
            # torch.save(self.optimizer.state_dict(), os.path.join(wandb.run.dir, 'models',f'{case}', f'optimizer_case_{case}_sl{slice}.pth'))

        if self.init_from_previous_slice and self.first:
            torch.save(self.NN_aif.state_dict(),
                       os.path.join(wandb.run.dir, 'models', f'{case}', f'aif_case_{case}_first.pth'))
            torch.save(self.NN_tissue.state_dict(),
                       os.path.join(wandb.run.dir, 'models', f'{case}', f'tac_case_{case}_first.pth'))
            torch.save(self.NN_ode.state_dict(),
                       os.path.join(wandb.run.dir, 'models', f'{case}', f'ode_case_{case}_first.pth'))


        # get results
        cbf = self.get_cbf(seconds=False).squeeze()
        mtt = self.get_mtt(seconds=True).squeeze()
        mtt_min = self.get_mtt(seconds=False).squeeze()
        delay = self.get_delay(seconds=True).squeeze()

        cbv = cbf * mtt_min
        tmax = delay + 0.5*mtt

        return {'cbf': cbf,
                'cbv': cbv,
                'mtt': mtt,
                'delay': delay,
                'tmax': tmax}

    def optimize(self,
                 batch_time,
                 batch_coordinates,
                 batch_aif,
                 batch_curves,
                 batch_boundary,
                 batch_collo,
                 batch_aif_time
                 ):

        batch_coordinates.requires_grad = True
        batch_collo.requires_grad = True

        self.train()
        self.optimizer.zero_grad()

        loss = torch.as_tensor(0.).to(self.device)
        loss_aif, loss_tissue, loss_residual = 999, 999, 999

        if self.lw_data:
            # compute data loss
            c_aif, c_tissue = self.forward_NNs(batch_aif_time, batch_coordinates)
            loss_aif, loss_tissue = self.__loss_data(batch_aif, batch_curves, c_aif, c_tissue)
            loss += self.lw_data * (loss_aif + loss_tissue)

        if self.lw_res:
            # compute residual loss
            c_aif, c_tissue, residual = self.forward_complete(batch_aif_time, batch_collo)
            loss_residual = self.__loss_residual(residual)
            loss += self.lw_res * loss_residual

        #     # compute data loss
        if np.isnan(float(loss.item())):
            raise ValueError('Loss is nan during training...')


        loss.backward()
        self.optimizer.step()

        batch_collo.requires_grad = False
        if not self.lw_res:
            loss_residual = torch.tensor(0)
        if not self.lw_data:
            loss_aif = torch.tensor(0)
            loss_tissue = torch.tensor(0)
        return loss_aif, loss_tissue, loss_residual

    def save_parameters(self):
        # Save NNs
        torch.save(self.state_dict(), os.path.join(wandb.run.dir, 'model.pth.tar'))
        torch.save(self.NN_tissue.state_dict(), os.path.join(wandb.run.dir, 'NN_tissue.pth.tar'))
        torch.save(self.NN_aif.state_dict(), os.path.join(wandb.run.dir, 'NN_aif.pth.tar'))
        # Save parameters
        torch.save(self.flow_mtt, os.path.join(wandb.run.dir, 'flow_mtt.pth.tar'))
        torch.save(self.flow_cbf, os.path.join(wandb.run.dir, 'flow_cbf.pth.tar'))
        torch.save(self.flow_t_delay, os.path.join(wandb.run.dir, 'flow_t_delay.pth.tar'))
        # Save parameter data
        for name, param in self.named_parameters():
            if 'flow_' in name:
                parameter_data = param.data.cpu().numpy()
                with open(os.path.join(wandb.run.dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, parameter_data)

        # Save parameter data
    def __loss_data(self, aif, curves, c_aif, c_tissue):
        # reshape the ground truth
        aif = aif.expand(*c_aif.shape)
        # solution loss
        loss_aif = F.mse_loss(aif, c_aif)
        loss_tissue = F.mse_loss(curves, c_tissue)
        return loss_aif, loss_tissue

    def __loss_interpolation(self, aif, curves, output):
        # TODO implement loss that uses C_aif(t-MTT) estimation and compares to interpolated version of AIF
        pass

    def __loss_residual(self, residual):
        loss_r = torch.mean(torch.square(residual))
        return loss_r

    def __loss_bc(self, output):
        _, _, _, _, _ = output
        # TODO implement
        loss_bc = 0
        return loss_bc

    def __fwd_gradients(self, ys, xs):
        v = torch.ones_like(ys)
        v.requires_grad = True
        g = torch.autograd.grad(
            outputs=[ys],
            inputs=xs,
            grad_outputs=[v],
            create_graph=True,
        )[0]
        w = torch.ones_like(g)
        w.requires_grad = True
        out = torch.autograd.grad(
            outputs=[g],
            inputs=v,
            grad_outputs=[w],
            create_graph=True,
        )[0]
        return out



def train(config):
    os.makedirs(os.path.join(wandb.run.dir, 'results', config.case), exist_ok=True)

    data_dict = load_ISLES_data(folder=r'data/ISLES2018',
                                case=config.case,
                                temporal_smoothing=config.temporal_smoothing,
                                dataset=config.dataset,
                                baseline_zero=config.baseline_zero)

    scan_dimensions = data_dict['curves'].shape[:-1]
    slices = scan_dimensions[0]

    cbf_results = np.zeros([*scan_dimensions], dtype=np.float32)
    cbv_results = np.zeros([*scan_dimensions], dtype=np.float32)
    mtt_results = np.zeros([*scan_dimensions], dtype=np.float32)
    delay_results = np.zeros([*scan_dimensions], dtype=np.float32)
    tmax_results = np.zeros([*scan_dimensions], dtype=np.float32)
    slice=slices//2
    set_seed(config['seed'])
    mask_data = data_dict['mask'][slice]
    valid_voxels = torch.where(mask_data == 1)
    shape_in = torch.Size([1, len(valid_voxels[0]), 1])
    ppinn = SPPINN(config,
                      shape_in=shape_in,
                      n_inputs=1,
                      std_t=data_dict['std_t'],
                      original_data_shape=scan_dimensions[1:],
                      original_indices=valid_voxels,
                      case=config.case,
                      slice=slice,
                      first=True
                      )

    _ = ppinn.fit(slice,
                            data_dict,
                            batch_size=config.batch_size,
                            epochs=int(config.epochs),
                            case=config.case)

    slice_list = slices * [0]
    l1 = list(reversed(range(slices//2)))
    l2 = list(range(slices//2, slices))
    if len(slice_list[0::2]) == len(l2) and len(slice_list[1::2]) == len(l1):
        slice_list[0::2] = l2
        slice_list[1::2] = l1
    else:
        slice_list[0::2] = l1
        slice_list[1::2] = l2

    for slice in tqdm(slice_list):
        set_seed(config['seed'])
        mask_data = data_dict['mask'][slice]
        valid_voxels = torch.where(mask_data == 1)
        shape_in = torch.Size([1, len(valid_voxels[0]), 1])
        if len(valid_voxels[0]) * data_dict['curves'].shape[-1] < config.batch_size:
            cbf_results[slice, ...] = np.zeros([256, 256])
            cbv_results[slice, ...] = np.zeros([256, 256])
            mtt_results[slice, ...] = np.zeros([256, 256])
            delay_results[slice, ...] = np.zeros([256, 256])
            tmax_results[slice, ...] = np.zeros([256, 256])
            continue
        ppinn = SPPINN(config,
                          shape_in=shape_in,
                          n_inputs=1,
                          std_t=data_dict['std_t'],
                          original_data_shape=scan_dimensions[1:],
                          original_indices=valid_voxels,
                          case=config.case,
                          slice=slice,
                          first=False
                          )

        result_dict = ppinn.fit(slice,
                                data_dict,
                                batch_size=config.batch_size,
                                epochs=int(config.epochs),
                                case=config.case)

        #
        mask_data = mask_data.cpu().numpy()
        for key in result_dict.keys():
            result_dict[key] = result_dict[key].cpu().detach().numpy()
            result_dict[key] *= mask_data

        result_dict = drop_unphysical_amc2(result_dict)

        cbf_results[slice, ...] = result_dict['cbf']
        cbv_results[slice, ...] = result_dict['cbv']
        mtt_results[slice, ...] = result_dict['mtt']
        delay_results[slice, ...] = result_dict['delay']
        tmax_results[slice, ...] = result_dict['tmax']

        visualize_amc(config.case, slice, result_dict, data_dict)
        # visualize_amc_sygno(case, slice, sygnovia_results, data_dict)

        # save maps as sitks
        save_perfusion_parameters_amc(config,
                                                 config.case,
                                                 cbf_results,
                                                 cbv_results,
                                                 mtt_results,
                                                 delay_results,
                                                 tmax_results,
                                                 data_dict
                                                )

if __name__ == "__main__":
    # parse the path of the json config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True, help='cuda usage')
    parser.add_argument('--gpu_device', type=int, default=0, help='GPU device')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--wandb_folder', type=str, default='./wandb/', help='wandb folder path')
    parser.add_argument('--wandb', type=bool, default=False, help='wandb usage')
    parser.add_argument('--mode', type=str, default='train', help='mode')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--n_units', type=int, default=16, help='number of units')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--init_from_previous_slice', type=bool, default=True, help='initialize from previous slice')
    parser.add_argument('--use_scheduler', type=bool, default=True, help='use scheduler')
    parser.add_argument('--lw_data', type=float, default=1, help='lw_data')
    parser.add_argument('--lw_res', type=float, default=1, help='lw_res')
    parser.add_argument('--lw_curves', type=float, default=1, help='lw_curves')
    parser.add_argument('--factor', type=int, default=1, help='factor')
    parser.add_argument('--milestone', type=int, default=200, help='milestone')
    parser.add_argument('--siren', type=bool, default=True, help='siren')
    parser.add_argument('--hidden_tissue', type=int, default=128, help='hidden tissue')
    parser.add_argument('--hidden_ode', type=int, default=64, help='hidden ode')
    parser.add_argument('--siren_w0', type=int, default=30, help='siren w0')
    parser.add_argument('--bn', type=bool, default=False, help='batch normalization')
    parser.add_argument('--batch_size', type=int, default=5000, help='batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs') # 100
    parser.add_argument('--filter_type', type=bool, default=False, help='filter type')
    parser.add_argument('--sd', type=int, default=0, help='standard deviation')
    parser.add_argument('--plot_params_every', type=int, default=1, help='plot params every')
    parser.add_argument('--log_domain', type=bool, default=True, help='log domain')
    parser.add_argument('--temporal_smoothing', type=bool, default=False, help='temporal smoothing')
    parser.add_argument('--baseline_zero', type=bool, default=True, help='baseline zero')
    parser.add_argument('--drop_edges', type=bool, default=True, help='drop edges')
    parser.add_argument('--drop_unphysical', type=bool, default=True, help='drop unphysical')


    parser.add_argument('--dataset', type=str, default='TRAINING', help='data')
    parser.add_argument('--case', type=str, default='case_14', help='data')
    config = parser.parse_args()
    # set environment variable for offline runs
    os.environ["WANDB_MODE"] = "online" if config.wandb else "offline"
    # Pass them to wandb.init
    wandb.init(config=config, project="SPPINN", allow_val_change=True)
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    set_seed(config['seed'])
    config['run_name'] = wandb.run.name
    config['run_id'] = wandb.run.id
    os.makedirs(os.path.join(wandb.run.dir, 'results'))

    pattern = re.compile("^case_\d{1,2}$")

    if config.case == 'all':
        if config.dataset == 'TRAINING':
            for case in tqdm(range(1, 95)):
                if case == 77:
                    continue
                config.update({'case': f'case_{case}'}, allow_val_change=True)
                train(config)
        else:
            for case in tqdm(range(1, 63)):
                config.update({'case': f'case_{case}'}, allow_val_change=True)
                train(config)
    elif pattern.match(config.case):
        train(config)
    else:
        raise ValueError('Please specify either a correct case number or "all"')

    print('Goodbye world!')