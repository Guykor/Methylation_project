import numpy as np
import pandas as pd


class ThetaGenerator:
    def __init__(self, tissues):
        self.tissues = tissues
        self.dim = len(tissues)
        self.rng = np.random.default_rng()
        self.params = None

    def __gen_theta(self, vec):
        vec = vec / vec.sum()
        df = pd.DataFrame(vec, index=self.tissues)
        df.index.rename('tissue', inplace=True)
        df.columns = ['value']
        return df

    def choice(self):
        res = np.zeros(self.dim)
        res[self.rng.choice(np.arange(self.dim))] = 1
        return self.__gen_theta(res)

    def beta(self, a, b):
        res = self.rng.beta(a, b, size=self.dim)
        return self.__gen_theta(res)

    def uniform(self):
        res = self.rng.uniform(0, 1, size=self.dim)
        return self.__gen_theta(res)

    def dirichlet(self, counts):
        assert len(counts) == self.dim
        res = self.rng.dirichlet(alpha=counts)
        return self.__gen_theta(res)

    def poisson(self, rep):
        lam = self.params['lam']
        if type(lam) is int:
            res = self.rng.poisson(lam, size=(self.dim, rep))
        else:  # list or int
            res = self.rng.poisson(lam, size=rep)
        return self.__gen_theta(res)
