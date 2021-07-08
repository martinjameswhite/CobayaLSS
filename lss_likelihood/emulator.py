import numpy as np
from scipy.special import expit
import json


class Emulator(object):

    def __init__(self, filebase, kmin=1e-3, kmax=1)
        super(Emulator, self).__init__()

        self.load(filebase)

        self.n_parameters = self.W[0].shape[0]
        self.n_components = self.W[-1].shape[-1]
        self.n_layers = len(self.W)
        self.nk = self.sigmas.shape[0]
        self.k = np.logspace(np.log10(kmin), np.log10(kmax), self.nk)

    def load(self, filebase):

        with open('{}_W.json'.format(filebase), 'r') as fp:
            self.W = json.load(fp)
            for i, wi in enumerate(self.W):
                self.W[i] = np.array(wi).astype(np.float32)

        with open('{}_b.json'.format(filebase), 'r') as fp:
            self.b = json.load(fp)
            for bi in self.b:
                bi = np.array(bi).astype(np.float32)

        with open('{}_alphas.json'.format(filebase), 'r') as fp:
            self.alphas = json.load(fp)
            for ai in self.alphas:
                ai = np.array(ai).astype(np.float32)

        with open('{}_betas.json'.format(filebase), 'r') as fp:
            self.betas = json.load(fp)
            for bi in self.betas:
                bi = np.array(bi).astype(np.float32)

        with open('{}_pc_mean.json'.format(filebase), 'r') as fp:
            self.pc_mean = np.array(json.load(fp)).astype(np.float32)

        with open('{}_pc_sigmas.json'.format(filebase), 'r') as fp:
            self.pc_sigmas = np.array(json.load(fp)).astype(np.float32)

        with open('{}_v.json'.format(filebase), 'r') as fp:
            self.v = np.array(json.load(fp)).astype(np.float32)

        with open('{}_sigmas.json'.format(filebase), 'r') as fp:
            self.sigmas = np.array(json.load(fp)).astype(np.float32)

        with open('{}_mean.json'.format(filebase), 'r') as fp:
            self.mean = np.array(json.load(fp)).astype(np.float32)

        with open('{}_fstd.json'.format(filebase), 'r') as fp:
            self.fstd = np.array(json.load(fp)).astype(np.float32)

        with open('{}_param_sigmas.json'.format(filebase), 'r') as fp:
            self.param_sigmas = np.array(json.load(fp)).astype(np.float32)

        with open('{}_param_mean.json'.format(filebase), 'r') as fp:
            self.param_mean = np.array(json.load(fp)).astype(np.float32)

    def activation(self, x, alpha, beta):
        return (beta + (expit(alpha * x) * (1 - beta))) * x

    def __call__(self, parameters):

        outputs = []
        x = (parameters - self.param_mean) * self.param_sigmas

        for i in range(self.n_layers - 1):

            # linear network operation
            x = x @ self.W[i] + self.b[i]

            # non-linear activation function
            x = self.activation(x, self.alphas[i], self.betas[i])

        # linear output layer
        x = ((x @ self.W[-1]) + self.b[-1]) * \
            self.pc_sigmas[:self.n_components] + \
            self.pc_mean[:self.n_components]
        x = np.sinh((x @ self.v[:, :self.n_components].T)
                    * self.sigmas + self.mean) * self.fstd

        return self.k, x
