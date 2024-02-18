import numpy as np
import os
import glob
import time

import ray
from ray.experimental import tqdm_ray

from .tools import invR_nonZero

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

        self.obs_case = OFCase(obs_case_dir)
        self.y_indexes = self.calc_y_indexes()
        self.case_path_list = self.case_dirs()
        self.n_menber = len(self.case_path_list)
        self.cases = self.__cases__()
        self.xa = np.empty([self.dim_emsemble, self.dim_x])
        self.xf = np.empty([self.dim_emsemble, self.dim_x])
        self.y0 = np.empty(len(self.y_indexes))
        self.H = self.createH()

    def calc_y_indexes(self):
        y0 = np.matlib.repmat(np.array(self.obs_cells), self.n_y_scaler, 1)
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
        xf = self.xf  # 20 x 30720
        H = self.H  # 30720
        nmem = self.dim_emsemble
        xfa = np.mean(xf, axis=0)
        dxf = xf - xfa
        dyf = (H @ xf.T - H @ xfa.reshape(-1, 1)).T
        y_indexes = self.y_indexes
        y0 = self.y0

        @ray.remote
        def xaj(j, args, bar):
            y_indexes, dyf, nmem, y0, H, xf, xfa, dxf = args
            invR, nzero = invR_nonZero(j, y_indexes)
            invR = invR[nzero][:, nzero]
            dyfj = dyf[:, nzero]
            C = np.dot(dyfj, invR)
            w, v = np.linalg.eig(np.identity(nmem) * (nmem - 1) + np.dot(C, dyfj.T))
            w = np.real(w)
            v = np.real(v)
            p_invsq = np.diag(1 / np.sqrt(w))
            p_inv = np.diag(1 / w)
            Wa = v @ p_invsq @ v.T
            Was = v @ p_inv @ v.T

            yHxf = y0[nzero] - (H @ xf.T).mean(axis=1)[nzero]
            xaj = xfa[j] + dxf[:, j] @ (Was @ C @ yHxf.T + np.sqrt(nmem - 1) * Wa)

            # for progress bar
            bar.update.remote(1)
            time.sleep(0.1)

            return xaj

        # for ray put
        ray.init(num_cpus=4)
        argset = [y_indexes, dyf, nmem, y0, H, xf, xfa, dxf]
        argset_ids = ray.put(argset)

        # for progress bar
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        bar = remote_tqdm.remote(total=self.dim_x)

        # parallel progress
        rayget = ray.get([xaj.remote(j, argset_ids, bar) for j in range(self.dim_x)])
        ray.shutdown()

        self.xa = np.array(rayget)

    def limit_alpha_in_xa(self):
        xa = self.xa
        xa_alpha = xa[:, 6 * 3072 : 7 * 3072]
        xa_alpha[xa_alpha < 0] = 0
        xa_alpha[xa_alpha > 1] = 1
        xa[:, 2 * 3072 : 3 * 3072] = 0  # Uza
        xa[:, 5 * 3072 : 6 * 3072] = 0  # Uzw
