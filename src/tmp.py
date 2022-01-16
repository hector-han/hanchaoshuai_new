import numpy as np
import logging
from fpa.fpa import FlowerPollinationAlgorithm


def fun_and_obsrv(x):
    f = np.sum(x * x)
    grad = 2 * x
    grad_norm = np.linalg.norm(grad)
    return [f, grad_norm]


logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
fpa_obj = FlowerPollinationAlgorithm(2, fun_and_obsrv, save_path='tmp.tsv')
fpa_obj.train()
