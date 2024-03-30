import numpy as np
import numpy.matlib as mb

import os
import glob

# import time


from .tools import cell_distance
from .tools import localizemat
from .tools import letkf_update
from .foamer import OFCase


class EnSim:
    def __init__(
        self,
        ensim_dir: str,
        prefix_sim_name: str,
        x_names: list,
        n_x_scaler: int,
        n_cells: int,
        dim_ensemble: int,
        y_names: list,
        n_y_scaler: int,
        obs_cells: list,
        obs_case_dir: str,
        num_cpus: int,
    ):
        self.ensim_dir = ensim_dir
        self.prefix_sim_name = prefix_sim_name
        self.x_names = x_names
        self.n_x_scaler = n_x_scaler
        self.n_cells = n_cells
        self.dim_x = n_cells * n_x_scaler
        self.dim_emsemble = dim_ensemble
        self.dim_y = len(obs_cells) * list(y_names)
        self.y_names = y_names
        self.n_y_scaler = n_y_scaler
        self.obs_cells = np.array(obs_cells)
        self.num_cpus = num_cpus

        self.obs_case = OFCase(obs_case_dir)
        self.y_indexes = self.calc_y_indexes()
        self.case_path_list = self.case_dirs()
        self.n_menber = len(self.case_path_list)
        self.cases = self.__cases__()
        self.xa = np.empty([self.dim_emsemble, self.dim_x])
        self.xf = np.empty([self.dim_emsemble, self.dim_x])
        self.y0 = np.empty(len(self.y_indexes))
        self.H = self.createH()
        self.mat_d = cell_distance(self.case_path_list[0])

    def calc_y_indexes(self):
        y0 = mb.repmat(np.array(self.obs_cells), self.n_y_scaler, 1)
        # ベクトル問題
        for i in range(self.n_y_scaler):
            y0[i] += i * self.n_cells
        return y0.reshape((1, -1))[0]

    def createH(self):
        H = np.identity(self.dim_x)
        return H[self.y_indexes]

    def case_dirs(self):
        like_dir = os.path.join(self.ensim_dir, self.prefix_sim_name) + "*"
        return glob.glob(like_dir)

    def __cases__(self):
        cases = []
        for cpath in self.case_path_list:
            cases.append(OFCase(cpath))
        return cases

    def bkup_time_dir(self, time_name: str, to_time_name: str):
        for i, case in enumerate(self.cases):
            case.copyTimeDir(time_name, to_time_name)

    def update_cases(self, time_name):
        for i, case in enumerate(self.cases):
            case.writeValues(self.xa[i], f"{time_name}", self.x_names)

    def ensemble_forcast(self, time_name):
        for i, case in enumerate(self.cases):
            case.forcast(f"{time_name}")
            self.xf[i] = case.getValues(time_name, self.x_names)

    def clearPatternInCases(self, pattern: str):
        for case in self.cases:
            case.clearPattern(pattern)

    def observation(self, time_name):
        case = self.obs_case
        self.y0 = case.getValues(time_name, self.y_names, self.obs_cells)

    def letkf_update(self):
        xf = self.xf
        H = self.H
        y_indexes = self.y_indexes
        y0 = self.y0
        lmat = localizemat(self.mat_d, 0.1)
        self.xa = letkf_update(xf, H, y0, y_indexes, lmat, self.num_cpus)

    def limit_alpha_in_xa(self):
        xa = self.xa
        xa_alpha = xa[:, 6 * 3072 : 7 * 3072]
        xa_alpha[xa_alpha < 0] = 0
        xa_alpha[xa_alpha > 1] = 1
        xa[:, 2 * 3072 : 3 * 3072] = 0  # Uza
        xa[:, 5 * 3072 : 6 * 3072] = 0  # Uzw
