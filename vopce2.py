# rqy

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class DataLoader(object):
    def __init__(self, filename):
        self.point_list = []
        self.data_len = 0
        with open(filename) as file_in:
            for line in file_in:
                self.data_len = self.data_len + 1
                b = line.split(' ')
                temp = np.zeros((1, 3))
                temp[0, 0] = float(b[0])
                temp[0, 1] = float(b[1])
                temp[0, 2] = float(b[2])
                self.point_list.append(temp)
        print('load data successfully, ' + str(self.data_len), 'points in total')

    def from_numpy(self, filename):
        temp = np.load(filename)
        self.point_list = []
        self.data_len = temp.shape[0]
        for i in range(temp.shape[0]):
            self.point_list.append(temp[i, :].reshape((1, 3)))

    def scale(self, ratio):
        for i in range(self.data_len):
            self.point_list[i][0, 0] = self.point_list[i][0, 0] * ratio
            self.point_list[i][0, 1] = self.point_list[i][0, 1] * ratio
            self.point_list[i][0, 2] = self.point_list[i][0, 2] * ratio

    def normalize(self):
        data = self.get_all_data()
        my_max = np.max(data, axis=0)
        my_min = np.min(data, axis=0)
        for i in range(self.data_len):
            self.point_list[i][0, 0] = (self.point_list[i][0, 0] - my_min[0]) / (my_max[0] - my_min[0])
            self.point_list[i][0, 1] = (self.point_list[i][0, 1] - my_min[1]) / (my_max[1] - my_min[1])
            self.point_list[i][0, 2] = (self.point_list[i][0, 2] - my_min[2]) / (my_max[2] - my_min[2])

    def sample_batch(self, batch_size):
        batch = random.sample(self.point_list, batch_size)
        temp = batch[0]
        for i in range(1, len(batch)):
            temp = np.vstack((temp, batch[i]))
        return temp

    def get_all_data(self):
        temp = self.point_list[0]
        for i in range(1, self.data_len):
            temp = np.vstack((temp, self.point_list[i]))
        return temp

    def save(self):
        data = self.get_all_data()
        np.save('loader.npy', data)

    def load(self):
        self.point_list = []
        data = np.load('loader.npy')
        for i in range(data.shape[0]):
            self.point_list.append(data[i, :].reshape((1, 3)))

class PermEqui2_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui2_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm = x.mean(0, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x

class DeepSet(nn.Module):
    def __init__(self, h_dim, x_dim, out_dim):
        super(DeepSet, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim

        self.phi = nn.Sequential(
            PermEqui2_mean(self.x_dim, self.h_dim),
            nn.Tanh(),
            PermEqui2_mean(self.h_dim, self.h_dim),
            nn.Tanh(),
            PermEqui2_mean(self.h_dim, self.h_dim),
            nn.Tanh(),
        )

        self.ro = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.Tanh(),
            nn.Linear(self.h_dim, out_dim),
        )

    def forward(self, x):
        phi_output = self.phi(x)
        sum_output, _ = phi_output.max(0)
        ro_output = self.ro(sum_output)
        return ro_output

class MDN(nn.Module):
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians

        self.mu_1 = nn.Linear(in_features, 256)
        self.mu_2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, 3 * num_gaussians)

        self.pi = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_gaussians),
            nn.Softmax(dim=1)
        )

        self.lam_1 = nn.Linear(in_features, 256)
        self.lam_2 = nn.Linear(256, 256)
        self.lam_3 = nn.Linear(256, 3 * num_gaussians)

        self.U_1 = nn.Linear(in_features, 256)
        self.U_2 = nn.Linear(256, 256)
        self.U_3 = nn.Linear(256, 9 * num_gaussians)

    def forward(self, x):
        pi = self.pi(x)
        mu = F.relu(self.mu_1(x))
        mu = F.relu(self.mu_2(mu))
        mu = self.mu(mu)
        mu = mu.view(-1, self.num_gaussians, 3)  # [1, K, 3]

        lam = F.relu(self.lam_1(x))
        lam = F.relu(self.lam_2(lam))
        lam = torch.exp(self.lam_3(lam))  # [1, K*3]
        lam = lam.view(-1, self.num_gaussians, 3)

        U = F.relu(self.U_1(x))
        U = F.relu(self.U_2(U))
        U = torch.exp(self.U_3(U))  # [1, K*3]
        U = U.view(-1, self.num_gaussians, 3, 3)
        return pi, mu, lam, U

    def gaussian_probability(self, mu, lam, U, target):
        N = target.size()[0]  # N = 2800
        # print('N', N)
        ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
        deter = torch.prod(lam, 2)  # deter = [1, 130]
        ret = torch.zeros(N, self.num_gaussians)
        # print('ret', ret.size())
        for j in range(self.num_gaussians):
            U_p = U[0, j, :, :]
            R, _ = torch.qr(U_p)
            eigen_mat = torch.diag_embed(lam[0, j, :] * lam[0, j, :])
            rot_eigen_mat = torch.mm(R, torch.mm(eigen_mat, torch.transpose(R, 0, 1)))  # [3, 3]
            b = torch.inverse(rot_eigen_mat)  # [3, 3]
            a = target - mu[0, j, :]
            c = a.t()
            # print('a', a.size())
            # print('b', b.size())
            # print('c', c.size())
            sigma_mat_rot = torch.diag(torch.mm(a, torch.mm(b, c)))  # [2800, 1]
            # print('sigma_mat_rot', sigma_mat_rot.size())
            prefix = ONEOVERSQRT2PI * ONEOVERSQRT2PI * ONEOVERSQRT2PI / deter[0, j]
            # print('prefix', prefix)
            temp = prefix * torch.exp(-0.5 * sigma_mat_rot)
            # print('temp', temp.size())
            ret[:, j] = temp
        return ret

    def gaussian_probability2(self, mu, lam, U, target):
        # mu torch.Size([1, 50, 3])
        # lam torch.Size([1, 50, 3])
        # U torch.Size([1, 50, 3, 3])

        N = target.size()[0]
        ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
        deter = torch.prod(lam, 2)

        # print(deter.size())
        ret = torch.zeros(N, self.num_gaussians)
        for i in range(N):
            data = target[i, :]
            for j in range(self.num_gaussians):
                a = (data - mu[0, j, :]).view(1, 3)
                c = a.t()
                eigen_mat = torch.diag_embed(lam[0, j, :] * lam[0, j, :])
                U_p = U[0, j, :, :]
                # print('U_p', U_p)
                R, _ = torch.qr(U_p)
                # R = self.gramschmidt(U_p)
                # w, v = torch.eig(R, eigenvectors=True)
                # print('R*R', torch.mm(torch.transpose(R, 0, 1), R))
                # print('Rw', w)
                rot_eigen_mat = torch.mm(R, torch.mm(eigen_mat, torch.transpose(R, 0, 1)))
                b = torch.inverse(rot_eigen_mat)
                # print('b', b)
                sigma_mat_rot = torch.mm(a, torch.mm(b, c))
                ret[i, j] = ONEOVERSQRT2PI * ONEOVERSQRT2PI * ONEOVERSQRT2PI / deter[0, j] * torch.exp(
                    -0.5 * sigma_mat_rot)
        return ret

    def mdn_loss1(self, pi, mu, sigma, rot, target):
        # pi torch.Size([1, 50])
        # mu torch.Size([1, 50, 3])
        # sigma torch.Size([1, 50, 3])
        # rot torch.Size([1, 50, 3])

        temp2 = self.gaussian_probability(mu, sigma, rot, target)
        prob = pi * temp2
        nll = -torch.log(torch.sum(prob, dim=1))
        return torch.mean(nll)

    def gramschmidt(self, A):
        """
        Applies the Gram-Schmidt method to A
        and returns Q and R, so Q*R = A.
        """
        R = torch.zeros(3, 3)
        Q = torch.zeros(3, 3)
        for k in range(0, 3):
            a = A[:, k].view(3, 1)
            R[k, k] = torch.sqrt(torch.mm(torch.transpose(a, 0, 1), a))[0, 0]
            Q[:, k] = A[:, k] / R[k, k]
            for j in range(k + 1, 3):
                q1 = Q[:, k].view(3, 1)
                a1 = A[:, j].view(3, 1)
                R[k, j] = torch.mm(torch.transpose(q1, 0, 1), a1)
                A[:, j] = A[:, j] - R[k, j] * Q[:, k]
        return Q

class VOPCE(nn.Module):
    def __init__(self, z_dim, x_dim, num_gauss):
        super(VOPCE, self).__init__()
        self.num_gauss = num_gauss
        self.z_dim = z_dim
        self.mdn = MDN(in_features=z_dim, out_features=x_dim, num_gaussians=num_gauss)
        self.deepset = DeepSet(h_dim=256, x_dim=x_dim, out_dim=z_dim*2)

    def encode(self, x):
        temp = self.deepset.forward(x).unsqueeze(0)
        z_mu = temp[0, 0:self.z_dim]
        log_std = temp[0, self.z_dim:2*self.z_dim]
        z_std = log_std.mul(0.5).exp_()
        esp = torch.randn(*z_mu.size())
        z = z_mu    #  + z_std * esp
        return z, z_mu, z_std

    def decode(self, z):
        pi, mu, sigma, rot = self.mdn.forward(z)
        return pi, mu, sigma, rot

    def forward(self, x):
        z, mu, std = self.encode(x)
        z = z.view(1, -1)
        pi, mu, sigma, rot = self.decode(z)
        return pi, mu, sigma, rot

    def clip_grad(self, model, max_norm):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
        total_norm = total_norm ** (0.5)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                p.grad.data.mul_(clip_coef)
        return total_norm

    def train(self, x):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        z, z_mu, z_std = self.encode(x)
        z = z.view(1, -1)
        pi, mu, sigma, rot = self.mdn.forward(z)
        loss1 = self.mdn.mdn_loss1(pi, mu, sigma, rot, x)
        loss2 = -0.5 * torch.mean(1 + z_std - z_mu.pow(2) - z_std.exp())
        loss = loss1 + loss2
        loss.backward()
        self.clip_grad(self, 1)
        optimizer.step()
        return loss.item()

    def save_model(self):
        torch.save(self.mdn, '130_gs_mdn_box_all.pkl')
        torch.save(self.deepset, '130_gs_deepset_box_all.pkl')

    def load_model(self):
        self.mdn = torch.load('130_gs_mdn_box_all.pkl')
        self.deepset = torch.load('130_gs_deepset_box_all.pkl')

    def sample_new(self, pi, mu, lam, U):
        # pi torch.Size([2816, 50])
        # sigma torch.Size([2816, 50, 3])
        # mu torch.Size([2816, 50, 3])
        # rot torch.Size([2816, 50, 3])
        categorical = Categorical(pi)
        pis = list(categorical.sample().data)
        point_list = []
        label = []
        for i in range(mu.size()[0]):   # N
            k = int(pis[i].item())
            eigen_mat = torch.diag_embed(lam[0, k, :] * lam[0, k, :])
            U_p = U[0, k, :, :]
            R, _ = torch.qr(U_p)
            rot_eigen_mat = torch.mm(R, torch.mm(eigen_mat, torch.transpose(R, 0, 1)))
            m = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu[i, k, :].view(3, ),
                                                                           covariance_matrix=rot_eigen_mat)
            point_list.append(m.sample())
            label.append(pis[i].item())
        sample = point_list[0].view(1, 3)
        for i in range(1, mu.size()[0]):
            temp = point_list[i].view(1, 3)
            sample = torch.cat((sample, temp), 0)
        return sample, label

    def sample(self, x, N):
        z, mu, std = self.encode(x)
        z = z.repeat(N, 1)
        pi, mu, lam, U = self.mdn.forward(z)
        x_pred, _ = self.sample_new(pi, mu, lam, U)
        return x_pred

if __name__ == '__main__':
    batch_size = 64
    my_loader = DataLoader('shapenet/1.txt')
    my_loader.from_numpy('fix_box.npy')
    vopce = VOPCE(z_dim=256, x_dim=3, num_gauss=130)
    my_loader.normalize()
    # vopce.load_model()

    # train
    loss_list = []
    for i in range(1000):
        # loss = vopce.train(torch.from_numpy(my_loader.sample_batch(batch_size)).float())
        loss = vopce.train(torch.from_numpy(my_loader.get_all_data()).float())
        if i % 1 == 0:
            print('iter', i, 'loss', loss)
            loss_list.append(loss)
            # vopce.save_model()

    curve = np.array(loss_list)
    # np.save('130_gs_box_all.npy', curve)
    # plt.plot(loss_list)'''

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    x = my_loader.get_all_data()
    # print(x.shape)
    x_list = []
    y_list = []
    z_list = []
    for i in range(my_loader.data_len):
        x_list.append(x[i, 0])
        y_list.append(x[i, 1])
        z_list.append(x[i, 2])
    ax1.scatter(x_list, y_list, z_list, marker='o')

    # rec_x = vopce.forward(torch.from_numpy(my_loader.get_all_data()).float()).detach().numpy()
    rec_x = vopce.sample(torch.from_numpy(my_loader.get_all_data()).float(), 2*my_loader.data_len)

    ax2 = fig.add_subplot(122, projection='3d')
    x_list = []
    y_list = []
    z_list = []
    for i in range(my_loader.data_len):
        x_list.append(rec_x[i, 0])
        y_list.append(rec_x[i, 1])
        z_list.append(rec_x[i, 2])
    ax2.scatter(x_list, y_list, z_list, marker='o')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_zlim(-0.1, 1.1)
    plt.show()
