import torch
import torch.nn as nn
import numpy as np
from models.MLP_st import MLP, MLP_ODE
from utils.train_utils import AverageMeter, set_seed
from utils.val_utils import load_phantom_gt, plot_results, log_software_results, drop_edges, drop_unphysical
from utils.data_utils import load_phantom_data

import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import wandb
import pickle
from einops.einops import rearrange
import argparse
class SPPINN(nn.Module):
    def __init__(self,
                 config,
                 data_dict):
        super(SPPINN, self).__init__()
        self.config = config
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
        # self.milestones = [self.config.epochs//3, 2*self.config.epochs//3]
        self.milestones = [400]
        self.perfusion_values = data_dict['perfusion_values']
        self.std_t = data_dict['std_t']
        self.neurons_out = 1
        # initialize flow parameters
        self.log_domain = config.log_domain
        self.flow_cbf = None
        self.flow_t_delay = None
        self.flow_mtt = None
        n_layers = config.n_layers
        n_units = config.n_units
        lr = config.lr
        loss_weights = (config.lw_data, config.lw_res, 0)
        bn = config.bn
        self.batch_size = config.batch_size
        self.data_coordinates_xy = None
        self.NN_aif = MLP(
            True,
            n_layers,
            n_units,
            n_inputs=1,
            neurons_out=1,
            bn=bn,
            act='tanh'
        )
       
        self.NN_tissue = MLP(
            False,
            n_layers,
            128,
            n_inputs=3,
            neurons_out=1,
            bn=bn,
            act='tanh'
        )

        self.NN_ode = MLP_ODE(
            n_layers,
            64,
            n_inputs=2,
            neurons_out=3,
            bn=bn,
            act='tanh'
        )

        self.current_iteration = 0
        self.epoch = 1
        self.set_lr(self.config.optimizer, lr)
        self.set_loss_weights(loss_weights)
        self.set_params_to_domain()
        self.set_device(self.device)
        self.float()

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
        # t = t.unsqueeze(-1)
        # steps = t.shape[0]
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

        c_aif_a = self.NN_aif(t - delay / self.std_t, xy) #128

        c_aif_b = self.NN_aif(t - delay / self.std_t - mtt / self.std_t, xy) #128


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
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=self.milestones,
                                                              gamma=0.5)

    def set_device(self, device):
        self.to(device)

    def set_params_to_domain(self):
        for name, param in self.named_parameters():
            if 'flow' in name:
                param.data = torch.log(param.data) if self.log_domain else param.data


    def get_ode_params(self):
        data_coordinates_xy = rearrange(self.data_coordinates_xy, 'dum1 dum2 x y val-> (dum1 dum2 x y) val')
        params = self.NN_ode(data_coordinates_xy)
        params = rearrange(params, '(dum1 dum2 x y) val -> dum1 dum2 x y val', dum1=1, dum2=1,
                                        x=224,
                                        y=224)
        self.flow_cbf = params[..., :1]
        self.flow_mtt = params[..., 1:2]
        self.flow_t_delay = params[..., 2:]

        if self.log_domain:
            # print(24 * torch.exp(self.flow_mtt))
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
        constant = torch.as_tensor(constant).to(self.device)
        flow, _, _ = self.get_ode_params()
        if seconds:
            return constant * flow
        else:
            return constant * flow * 60
    # @profile
    def fit(self, data_dict):
        gt = data_dict['perfusion_values']
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        data_time = data_dict['time'].to(self.device)
        data_aif = data_dict['aif'].to(self.device)
        data_curves = data_dict['curves'].numpy()
        data_coordinates = data_dict['coordinates'].numpy()
        self.data_coordinates_xy = data_dict['coordinates_xy_only'].to(self.device)

        timepoints = len(data_dict['aif'])

        collocation_txys = np.zeros((1,1, 224, 224, timepoints, 3))
        collocation_txys[...,1:] = data_coordinates[...,1:]

        data_curves = rearrange(data_curves, 'dum1 dum2 x y (t val)-> (dum1 dum2 x y t) val', val=1)
        data_coordinates = rearrange(data_coordinates, 'dum1 dum2 x y t val-> (dum1 dum2 x y t) val')
        collocation_coordinates = rearrange(collocation_txys, 'dum1 dum2 x y t val -> (dum1 dum2 x y t) val')

        for ep in tqdm(range(self.current_iteration + 1, self.current_iteration + epochs + 1)):
            collocation_coordinates[:, 0] = torch.FloatTensor(*collocation_coordinates.shape[:-1]).uniform_(
                data_dict['coll_points_min'],
                data_dict['coll_points_max'])
            integers = np.arange(len(data_curves))
            np.random.shuffle(integers)
            splits = np.array_split(integers, int(len(data_curves)/batch_size))

            epoch_aif_loss = AverageMeter()
            epoch_tissue_loss = AverageMeter()
            epoch_residual_loss = AverageMeter()

            for split in splits:

                batch_curves = torch.from_numpy(data_curves[split]).float().to(self.device)
                batch_coordinates = torch.from_numpy(data_coordinates[split]).float().to(self.device)
                batch_collo = torch.from_numpy(collocation_coordinates[split]).float().to(self.device)

                batch_aif = data_aif
                batch_time = data_time[:,0,0,:]
                loss_aif, loss_tissue, loss_residual = self.optimize(batch_time,
                                                                     batch_coordinates,
                                                                     batch_aif,
                                                                     batch_curves,
                                                                     batch_collo)
                epoch_aif_loss.update(loss_aif.item())
                epoch_tissue_loss.update(loss_tissue.item())
                epoch_residual_loss.update(loss_residual.item())

            self.scheduler.step()

            if self.config.wandb:
                metrics = {"aif_loss": epoch_aif_loss.avg,
                           "tissue_loss": epoch_tissue_loss.avg,
                           "residual_loss": epoch_residual_loss.avg,
                           "lr": self.optimizer.param_groups[0]['lr'],
                           }
                validation_metrics = self.validate()
                metrics.update(validation_metrics)

                wandb.log(metrics)

            if self.epoch % self.config.plot_params_every == 0:
                try:
                    self.plot_params(0, 0, gt, ep)
                except:
                    continue

            self.current_iteration += 1
            self.epoch += 1
    # @profile
    def optimize(self,
                 batch_time,
                 batch_coordinates,
                 batch_aif,
                 batch_curves,
                 batch_collo):

        batch_coordinates.requires_grad = True
        batch_collo.requires_grad = True

        self.train()
        self.optimizer.zero_grad()

        loss = torch.as_tensor(0.).to(self.device)
        loss_aif, loss_tissue, loss_residual = 999, 999, 999

        if self.lw_data:
            # compute data loss
            c_aif, c_tissue = self.forward_NNs(batch_time, batch_coordinates)
            loss_aif, loss_tissue = self.__loss_data(batch_aif, batch_curves, c_aif, c_tissue)

            loss += self.lw_data * (loss_aif + loss_tissue)

        if self.lw_res:
            # compute residual loss

            c_aif, c_tissue, residual = self.forward_complete(batch_time, batch_collo)

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
        return loss_aif, loss_tissue, loss_residual

    def get_results(self, save_results=True, st=False):
        # self.ema_on = True
        cbf = self.get_cbf(seconds=False).squeeze().cpu().detach().numpy()
        mtt = self.get_mtt(seconds=True).squeeze().cpu().detach().numpy()
        mtt_min = self.get_mtt(seconds=False).squeeze().cpu().detach().numpy()
        delay = self.get_delay(seconds=True).squeeze().cpu().detach().numpy()
        cbv = cbf * mtt_min
        tmax = delay + 0.5 * mtt
        results_dict = {'cbf': cbf, 'cbv': cbv, 'mtt': mtt, 'delay': delay, 'tmax': tmax}
        if save_results:
            if not st:
                with open(os.path.join(wandb.run.dir, f'ppinn_results_cbv_{self.config.cbv_ml}_sd_{self.config.sd}_undersample_{self.config.undersampling}.pickle'), 'wb') as f:
                    pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(os.path.join(wandb.run.dir, f'sppinn_results_cbv_{self.config.cbv_ml}_sd_{self.config.sd}_undersample_{self.config.undersampling}.pickle'), 'wb') as f:
                    pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return results_dict

    def validate(self):

        # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'

        gt_cbv = self.perfusion_values[..., 0]
        gt_delay = self.perfusion_values[..., 1]
        gt_mtt = self.perfusion_values[..., 2] * 60
        gt_cbf = self.perfusion_values[..., 3]

        [gt_cbv, gt_cbf, gt_mtt, gt_delay] = [torch.as_tensor(x).to(self.device)
                                              for x in [gt_cbv, gt_cbf, gt_mtt, gt_delay]]
        cbf = self.get_cbf(seconds=False).squeeze(-1)
        mtt = self.get_mtt(seconds=True).squeeze(-1)
        mtt_min = self.get_mtt(seconds=False).squeeze(-1)
        delay = self.get_delay(seconds=True).squeeze(-1).squeeze(-1)
        cbv = cbf * mtt_min
        cbv_mse = torch.nn.functional.mse_loss(cbv, gt_cbv).item()
        cbf_mse = torch.nn.functional.mse_loss(cbf, gt_cbf).item()
        mtt_mse = torch.nn.functional.mse_loss(mtt, gt_mtt).item()
        delay_mse = torch.nn.functional.mse_loss(delay, gt_delay).item()
        return {'cbv_mse': cbv_mse,
                'cbf_mse': cbf_mse,
                'mtt_mse': mtt_mse,
                'delay_mse': delay_mse}

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

    def plot_params(self, i, j, perfusion_values, epoch):
        cbf = self.get_cbf(seconds=False).squeeze(-1)
        mtt = self.get_mtt(seconds=True).squeeze(-1)
        mtt_min = self.get_mtt(seconds=False).squeeze(-1)
        delay = self.get_delay(seconds=True).squeeze(-1)
        # cbf = torch.clip(cbf, min=0, max=125)
        cbv = cbf * mtt_min
        # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
        gt_cbv = perfusion_values[..., 0]
        gt_delay = perfusion_values[..., 1]
        gt_mtt = perfusion_values[..., 2] * 60
        gt_cbf = perfusion_values[..., 3]

        cbf_min, cbf_max = 0.9 * torch.min(gt_cbf).item(), 1.1 * torch.max(gt_cbf).item()

        [cbf, mtt, cbv, gt_cbf, gt_mtt, gt_cbv, delay] = [x.detach().cpu().numpy() for x in
                                                          [cbf, mtt, cbv, gt_cbf, gt_mtt, gt_cbv, delay]]
        i, j = 0, 0

        font = {'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 15,
                }
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["figure.dpi"] = 150
        fig, ax = plt.subplots(3, 4, figsize=(10, 12))

        ax[0, 0].set_title('CBF', fontdict=font)
        ax[0, 0].imshow(cbf[i, j], vmin=cbf_min, vmax=cbf_max, cmap='jet')
        im = ax[1, 0].imshow(gt_cbf[i, j], vmin=cbf_min, vmax=cbf_max, cmap='jet')
        cax = ax[2, 0].inset_axes([0, 0.82, 1, 0.1])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal")
        bar.outline.set_color('black')
        bar.set_label('ml/100g/min', fontdict=font)
        bar.ax.tick_params(labelsize=14)
        ax[0, 0].set_ylabel('PPINN', fontdict=font)
        ax[1, 0].set_ylabel('GT', fontdict=font)

        ax[0, 1].set_title('MTT (s)', fontdict=font)
        ax[0, 1].imshow(mtt[i, j], vmin=0.01, vmax=1.1 * 24, cmap='jet')
        im = ax[1, 1].imshow(gt_mtt[i, j], vmin=0.01, vmax=1.1 * 24, cmap='jet')
        cax = ax[2, 1].inset_axes([0, 0.82, 1, 0.1])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal")
        bar.outline.set_color('black')
        bar.set_label('seconds', fontdict=font)
        bar.ax.tick_params(labelsize=14)

        ax[0, 2].set_title('CBV (ml/100g)', fontdict=font)
        ax[0, 2].imshow(cbv[i, j], vmin=0.01, vmax=7, cmap='jet')
        im = ax[1, 2].imshow(gt_cbv[i, j], vmin=0.01, vmax=7, cmap='jet')
        cax = ax[2, 2].inset_axes([0, 0.82, 1, 0.1])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal")
        bar.outline.set_color('black')
        bar.set_label('ml/100g', fontdict=font)
        bar.ax.tick_params(labelsize=14)

        ax[0, 3].set_title('Delay (s)', fontdict=font)
        ax[0, 3].imshow(delay[i, j], vmin=0.01, vmax=3.5, cmap='jet')
        im = ax[1, 3].imshow(gt_delay[i, j], vmin=0.01, vmax=3.5, cmap='jet')
        cax = ax[2, 3].inset_axes([0, 0.82, 1, 0.1])
        bar = fig.colorbar(im, cax=cax, orientation="horizontal")
        bar.outline.set_color('black')
        bar.set_label('seconds', fontdict=font)
        bar.ax.tick_params(labelsize=14)

        for i in range(4):
            ax[2, i].set_axis_off()
        for x in ax.flatten():
            x.axes.xaxis.set_ticks([])
            x.axes.yaxis.set_ticks([])
        fig.suptitle('Parameter estimation epoch: {}'.format(epoch), fontdict=font)
        plt.tight_layout()
        wandb.log({"parameters": plt}, step=epoch)
        plt.close()

def train(config):
    data_dict = load_phantom_data(gaussian_filter_type=config.filter_type,
                                     sd=config.sd,
                                     cbv_ml=config.cbv_ml,
                                     simulation_method=config.simulation_method,
                                     temporal_smoothing=config.temporal_smoothing,
                                     baseline_zero=config.baseline_zero)
    sppinn = SPPINN(config, data_dict)
    sppinn.fit(data_dict)
    sppinn.save_parameters()
    ppinn_results = sppinn.get_results(st=True)
    return ppinn_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True, help='cuda usage')
    parser.add_argument('--gpu_device', type=int, default=0, help='GPU device')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--wandb_folder', type=str, default='./wandb/', help='wandb folder path')
    parser.add_argument('--wandb', type=bool, default=False, help='wandb usage')
    parser.add_argument('--mode', type=str, default='train', help='mode')
    parser.add_argument('--cbv_ml', type=int, default=4, help='cbv_ml')
    parser.add_argument('--simulation_method', type=int, default=2, help='simulation method')
    parser.add_argument('--n_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--n_units', type=int, default=16, help='number of units')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lw_data', type=float, default=1, help='lw_data')
    parser.add_argument('--lw_res', type=float, default=1, help='lw_res')
    parser.add_argument('--lw_curves', type=float, default=1, help='lw_curves')
    parser.add_argument('--factor', type=int, default=1, help='factor')
    parser.add_argument('--ode_net_type', type=str, default='MLP_tanh', help='ode net type')
    parser.add_argument('--milestone', type=int, default=100, help='milestone')
    parser.add_argument('--bn', type=bool, default=False, help='bn')
    parser.add_argument('--batch_size', type=int, default=5000, help='batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--siren', type=bool, default=False, help='siren')
    parser.add_argument('--siren_w0', type=int, default=30, help='siren w0')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--filter_type', type=str, default='gauss_spatial', help='filter type')
    parser.add_argument('--sd', type=int, default=2, help='sd')
    parser.add_argument('--plot_params_every', type=int, default=1, help='plot params every')
    parser.add_argument('--log_domain', type=bool, default=True, help='log domain')
    parser.add_argument('--temporal_smoothing', type=bool, default=True, help='temporal smoothing')
    parser.add_argument('--baseline_zero', type=bool, default=True, help='baseline zero')
    parser.add_argument('--drop_edges', type=bool, default=True, help='drop edges')
    parser.add_argument('--drop_unphysical', type=bool, default=True, help='drop unphysical')
    parser.add_argument('--data', type=str, default='phantom', help='data')
    parser.add_argument('--undersampling', type=float, default=0.0, help='undersampling')

    config = parser.parse_args()

    # set environment variable for offline runs
    os.environ["WANDB_MODE"] = "online" if config.wandb else "offline"
    # Pass them to wandb.init
    wandb.init(config=config, project="SPPINN")
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    set_seed(config['seed'])
    config['run_name'] = wandb.run.name
    config['run_id'] = wandb.run.id
    os.makedirs(os.path.join(wandb.run.dir, 'results'))

    if config.data == 'phantom':
        results = {
            'gt': load_phantom_gt(cbv_ml=config.cbv_ml),
            'sppinn': train(config)
        }

        plot_results(results)
        log_software_results(results, config.cbv_ml)
        results = drop_edges(results)  # if config.drop_edges else results
        results = drop_unphysical(results)  # if config.drop_unphysical else results
        plot_results(results, corrected=True)
        log_software_results(results, config.cbv_ml, corrected=True)

