import numpy as np

class AO:
    def __init__(self, h, H, g):
        self.h = h
        self.H = H
        self.g = g
        self.num = 1e3
        self.e = 1e-9

    def compute(self):
        habs =  np.linalg.norm(self.h, axis=1).reshape(-1,1)
        w = self.h / habs
        for k in self.num:
            pass


    




