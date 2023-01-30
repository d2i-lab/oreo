import random
import sys
import numpy as np


def swap(A, pi):
    idx = np.random.choice(range(len(A)), 1)[0]
    pi_new = [A[idx]]
    A = np.concatenate([A[:idx], A[idx + 1:], pi])
    return A, pi_new


def move(A, pi):
    idx = np.random.choice(range(len(A)), 1)[0]
    pi_new = [A[idx]]
    A = np.concatenate([A[:idx], A[idx+1:]])
    return A, pi_new


def downsample(A, pi, c, c_new):
    u = np.random.uniform()
    frac_c = c - int(c)
    A_new = A
    pi_new = []
    if int(c_new) == 0:
        if u > frac_c / c:
            A_new, pi_new = swap(A, pi)
        A_new = []
    elif 0 < int(c_new) == int(c):
        if u > (1 - (c_new / c) * frac_c) / (1 - frac_c):
            A_new, pi_new = swap(A, pi)
    else:
        if u <= (c_new / c) * frac_c:
            A_new = np.random.choice(A, min(int(c_new), len(A)), replace=False)
            A_new, pi_new = swap(A_new, pi)
        else:
            A_new = np.random.choice(A, min(int(c_new) + 1, len(A)), replace=False)
            A_new, pi_new = move(A_new, pi)

    if c_new == int(c_new):
        pi_new = []

    return A_new, pi_new


def s_round(x):
    a = int(x)
    b = a + 1
    return np.random.choice([a, b], p=[b - x, x - a])


class WeightedReservoir:
    def __init__(self, k=1000, batch_size=100, policy=["res", "sw"]):
        self.k = k
        self.w = batch_size
        self.policy = policy
        min_gamma = 1 - batch_size * 1. / k
        self.gammas = np.linspace(1, min_gamma, 3)[1:2]
        self.pools = []
        self.partials = []
        self.cw = []
        self.sliding_window = []
        np.random.seed(0)
        for g in self.gammas:
            self.pools.append([])
            self.partials.append([])
            self.cw.append(0)

    def _update_sw(self, items):
        # Update sliding window
        self.sliding_window.extend(items)
        N = len(self.sliding_window)
        if N > self.w:
            extra = N - self.w
            self.sliding_window = self.sliding_window[extra:]

    def _update_res(self, items):
        b = len(items)
        # Update reservoir sample
        for i, gamma in enumerate(self.gammas):
            A = self.pools[i]
            pi = self.partials[i]
            C = self.cw[i]
            W = self.cw[i]
            if self.cw[i] < self.k:
                W *= gamma
                if W > 0:
                    A, pi = downsample(A, pi, C, W)
                A = np.concatenate([A, items])
                W += b
                if W > self.k:
                    A, pi = downsample(A, pi, W, self.k)
            else:
                W = W * gamma + b
                if W >= self.k:
                    m = s_round(b * 1. * self.k / W)
                    if m > 0:
                        idx1 = np.random.choice(len(A), m, replace=False)
                        idx2 = np.random.choice(b, m, replace=False)
                        for j in range(m):
                            A[idx1[j]] = items[idx2[j]]
                else:
                    A, pi = downsample(A, pi, self.k, W - b)
                    A = np.concatenate([A, items])
            self.cw[i] = W
            self.pools[i] = A
            self.partials[i] = pi

    def insert(self, items):
        self._update_res(items)
        if "sw" in self.policy:
            self._update_sw(items)

    def get_sw(self):
        return [self.sliding_window]

    def get_res(self):
        samples = []
        for i in range(len(self.gammas)):
            c = min(self.cw[i], self.k)
            frac_c = c - int(c)
            A = list(self.pools[i])
            # Fractional sample
            if np.random.uniform() < frac_c:
                A.extend(self.partials[i])
            samples.append(A)
        return samples

    def get_samples(self, kind):
        if 'sw' in kind:
            return self.get_sw()
        else:
            return self.get_res()

    def get_profile(self):
        return self.pools[0]

