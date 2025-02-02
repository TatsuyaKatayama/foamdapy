import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import ray
from PyFoam.RunDictionary.ParsedParameterFile import Field, ParsedParameterFile

from .tools import parallel_run


class ColumnInfo:

    def __init__(self,
                 name: str,
                 time_name: str,
                 indices: Union[List[int], slice],
                 cells: List[int] = None):
        # 列情報を初期化する
        self.name = name  # 列の名前
        self.time_name = time_name  # 時間ディレクトリの名前
        self.indices = indices  # 列のインデックス（リストまたはスライス）
        self.cells = cells  # セル指定あれば

    def str_dump(self):
        dic = {
            "name": self.name,
            "time_name": self.time_name,
            "indices": str(self.indices),
            "cells": self.cells.tolist() if self.cells is not None else None
        }
        return json.dumps(dic)


class esarray(np.ndarray):

    def __new__(cls, input_array, column_info: Dict[str, Union[List[int], slice]] = None):
        # 新しいesarrayオブジェクトを作成する
        obj = np.asarray(input_array).view(cls)
        obj.column_info = column_info or {}  # 列情報を初期化（指定がなければ空の辞書）
        return obj

    def __init__(self, input_array, column_info: dict = {}):
        # __init__メソッドは通常の初期化処理を行います
        self.column_info = column_info

    def __array_finalize__(self, obj):
        # ndarrayのサブクラス化に必要なメソッド
        if obj is None:
            return
        self.column_info = getattr(obj, 'column_info', {})

    @property
    def is_ensemble(self):
        return self.ndim == 2

    def _load_from_of(self, case_path: str, column_name: str) -> 'esarray':
        # 指定された列のデータをファイルから読み込む
        if column_name not in self.column_info:
            raise KeyError(f"列 '{column_name}' が見つかりません")
        colminfo = self.column_info[column_name]
        time_name = colminfo.time_name
        xname = colminfo.name
        cells = colminfo.cells

        Xfile = ParsedParameterFile(os.path.join(case_path, time_name, xname))
        Xai = np.array(Xfile.content["internalField"])

        # slice with cells
        if cells is not None:
            Xai = Xai[cells]
        return Xai

    def all_load(self, column_name: str, case_dirs: List[str], num_cpus: int = 1) -> 'esarray':
        if column_name not in self.column_info:
            raise KeyError(f"列 '{column_name}' が見つかりません")

        col_indices = self.column_info[column_name].indices

        if num_cpus == 1:
            for i, case_path in enumerate(case_dirs):
                self[i, col_indices] = self._load_from_of(case_path, column_name)
            return self

        ray_get = ray.get([
            parallel_run.remote(self._load_from_of, case_path, column_name) for case in self.cases
        ])

        return self.update(column_name, np.array(ray_get))

    def update(self, column_name: str, source: np.ndarray) -> 'esarray':
        # 指定された列のデータを更新する
        if column_name not in self.column_info:
            raise KeyError(f"列 '{column_name}' が見つかりません")
        indices = self.column_info[column_name].indices
        self[:, indices] = source
        return self

    def pickup(self, column_name: str) -> np.ndarray:
        # 指定された列のデータを抽出する
        if column_name not in self.column_info:
            raise KeyError(f"列 '{column_name}' が見つかりません")
        indices = self.column_info[column_name].indices
        return self[:, indices]

    def _save_to_of(self, args, column_name: str):
        i, case_path = args
        # 指定された列のデータをファイルから読み込む
        if column_name not in self.column_info:
            raise KeyError(f"列 '{column_name}' が見つかりません")
        colminfo = self.column_info[column_name]
        time_name = colminfo.time_name
        xname = colminfo.name
        cells = colminfo.cells
        indices = colminfo.indices
        Xa = self[i, indices]

        Xfile = ParsedParameterFile(os.path.join(case_path, time_name, xname))
        Xc = Xfile.content["internalField"]

        # internalFieldの値作成
        if "vector" in Xc.name:
            Xa = Xa.reshape([3, -1]).T

        # 書き戻し
        fi = Field(Xa.tolist(), Xc.name)
        p = re.compile("\n  3\n")
        Xfile.content["internalField"] = p.sub("", str(fi))
        Xfile.writeFile()

    def all_save(self, column_name: str, case_dirs: List[str], num_cpus: int = 1):
        if column_name not in self.column_info:
            raise KeyError(f"列 '{column_name}' が見つかりません")

        col_indices = self.column_info[column_name].indices

        if num_cpus == 1:
            for i, case_path in enumerate(case_dirs):
                self._save_to_of((i, case_path), column_name)
            return

        [
            parallel_run.remote(self._save_to_of, (i, case_path), column_name)
            for i, case_path in enumerate(case_dirs)
        ]

    def add_column_info(self,
                        key: str,
                        name: str,
                        time_name: str,
                        columns: Union[List[int], slice, Tuple[int, int]],
                        cells: List[int] = None):
        # 新しい列情報を追加する
        if isinstance(columns, (list, slice)):
            indices = columns
        elif isinstance(columns, tuple) and len(columns) == 2:
            indices = slice(*columns)
        else:
            raise ValueError("columns は整数のリスト、スライスオブジェクト、または2つの整数のタプルである必要があります")

        self.column_info[key] = ColumnInfo(name, time_name, indices, cells)

    def remove_column_info(self, name: str):
        # 指定された列情報を削除する
        if name in self.column_info:
            del self.column_info[name]
