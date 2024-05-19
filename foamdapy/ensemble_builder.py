import os
import shutil
import numpy as np

from .tools import decimal_normalize
from .foamer import OFCase


class ensemble_case_builder:
    def __init__(
        self, org_case_dir: str, ensemble_dir: str, ensemble_prefix: str = "ofsim_"
    ):
        self.org_case = OFCase(org_case_dir)
        self.ensemble_dir = ensemble_dir
        self.ensemble_prefix = ensemble_prefix

    def calc_org(self, end_time: str, writeInterval: str = None):
        self.org_case.forcast(end_time, writeInterval)

    def add_menber(self, new_menber_dir: str, clone_dirs_in_org: list):
        for dir in clone_dirs_in_org:
            org_dir = os.path.join(self.org_case.case_dir, dir)
            new_dir = os.path.join(new_menber_dir, dir)
            shutil.copytree(org_dir, new_dir, True)

    def rename_dir(self, in_dir: str, oldname: str, newname: str):
        olddir = os.path.join(in_dir, oldname)
        newdir = os.path.join(in_dir, newname)
        shutil.move(olddir, newdir)

    def create_member(self, ti: float, tj: float, dt: float, t_en: float):
        for i, t in enumerate(np.arange(ti, tj, dt)):
            new_menber = self.ensemble_prefix + f"{i}".rjust(2, "0")
            new_menber_dir = os.path.join(self.ensemble_dir, new_menber)
            timename = decimal_normalize(t)
            clone_dirs_in_org = ["0", "constant", "system", timename]

            self.add_menber(new_menber_dir, clone_dirs_in_org)
            self.rename_dir(new_menber_dir, timename, decimal_normalize(t_en))

    def allrun(self, ti: float, tj: float, dt: float, t_en: float):
        self.calc_org(f"{ti}")
        self.calc_org(f"{tj}", f"{dt}")
        self.create_member(ti, tj, dt, t_en)
