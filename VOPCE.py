import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from myLoader import MyLoader
from torch.distributions import Categorical
import math
import torch.nn.functional as F

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
        self.pi = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.mu_1 = nn.Linear(in_features, 256)
        self.mu_2 = nn.Linear(256, 256)
        self.sigma_1 = nn.Linear(in_features, 256)
        self.sigma_2 = nn.Linear(256, 256)
        self.sigma = nn.Linear(256, out_features * num_gaussians)
        self.mu = nn.Linear(256, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = F.relu(self.sigma_1(minibatch))
        sigma = F.relu(self.sigma_2(sigma))
        sigma = torch.exp(self.sigma(sigma))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = F.relu(self.mu_1(minibatch))
        mu = F.relu(self.mu_2(mu))
        mu = self.mu(mu)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


    def gaussian_probability(self, sigma, mu, target):
        _, k, _ = sigma.size()
        target = target.unsqueeze(1).repeat(1, k, 1)
        ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / sigma
        return torch.prod(ret, 2)

    def mdn_loss(self, pi, sigma, mu, target):
        prob = pi * self.gaussian_probability(sigma, mu, target)
        nll = -torch.log(torch.sum(prob, dim=1))
        return torch.mean(nll)

    def sample(self, pi, sigma, mu):
        categorical = Categorical(pi)
        pis = list(categorical.sample().data)
        sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
        for i, idx in enumerate(pis):
            sample[i] = sample[i].mul(sigma[i, idx]).add(mu[i, idx])
        return sample

class VOPCE(nn.Module):
    def __init__(self, z_dim, x_dim, num_gauss):
        super(VOPCE, self).__init__()
        self.z_dim = z_dim
        self.mdn = MDN(in_features=z_dim, out_features=x_dim, num_gaussians=num_gauss)
        self.deepset = DeepSet(h_dim=256, x_dim=x_dim, out_dim=z_dim*2)

    def encode(self, x):
        temp = self.deepset.forward(x).unsqueeze(0)
        z_mu = temp[0, 0:self.z_dim]
        log_std = temp[0, self.z_dim:2*self.z_dim]
        z_std = log_std.mul(0.5).exp_()
        esp = torch.randn(*z_mu.size())
        z = z_mu + z_std * esp
        return z, z_mu, z_std

    def decode(self, z):
        pi, sigma, mu = self.mdn.forward(z)
        x_pred = self.mdn.sample(pi, sigma, mu)
        return x_pred

    def forward(self, x):
        N = len(x)
        z, mu, std = self.encode(x)
        z = z.repeat(N, 1)
        x_pred = self.decode(z)
        return x_pred

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
        N = len(x)
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        z, z_mu, z_std = self.encode(x)
        # print('z', z[0, 0])
        z = z.repeat(N, 1)
        pi, sigma, mu = self.mdn.forward(z)
        # print(pi)
        loss1 = self.mdn.mdn_loss(pi, sigma, mu, x)
        loss2 = -0.5 * torch.mean(1 + z_std - z_mu.pow(2) - z_std.exp())
        # print('loss2', loss2)
        loss = loss1 + loss2
        loss.backward()
        print(loss)
        self.clip_grad(self, 1)
        optimizer.step()

    def save_model(self):
        torch.save(self, 'VOPCE.pkl')

    def load_model(self):
        self.vae = torch.load('VOPCE.pkl')

if __name__ == '__main__':
    n_sample = 1024
    my_loader = MyLoader('ModelNet', n_sample=n_sample)
    my_vopce = VOPCE(z_dim=50, x_dim=3, num_gauss=50)
    render_list = []
    render_obj_1 = my_loader.get_obj(index=0, color=[1.0, 0.0, 1.0], pos=[-1.5, 1.0, -6.0], rot=[0.0, 0.0, 0.0])
    render_list.append(render_obj_1)
    render_obj_2 = my_loader.get_obj(index=1, color=[1.0, 0.0, 1.0], pos=[-1.5, -1.0, -6.0], rot=[0.0, 0.0, 0.0])
    render_list.append(render_obj_2)

    # train
    for i in range(1000):
        print('iter', i)
        my_vopce.train(torch.from_numpy(render_obj_1.data).float())
        my_vopce.train(torch.from_numpy(render_obj_2.data).float())

    # reconstruct x
    x1 = torch.from_numpy(render_obj_1.data).float()
    y1 = my_vopce.forward(x1).detach().numpy()
    x2 = torch.from_numpy(render_obj_2.data).float()
    y2 = my_vopce.forward(x2).detach().numpy()

    render_obj_3 = my_loader.get_obj(index=0, color=[1.0, 1.0, 0.0], pos=[1.5, 1.0, -6.0], rot=[0.0, 0.0, 0.0])
    render_obj_3.data = y1
    render_list.append(render_obj_3)
    render_obj_4 = my_loader.get_obj(index=1, color=[1.0, 1.0, 0.0], pos=[1.5, -1.0, -6.0], rot=[0.0, 0.0, 0.0])
    render_obj_4.data = y2
    render_list.append(render_obj_4)
    my_loader.plot(render_list)
