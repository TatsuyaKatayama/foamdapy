import os
import numpy as np
import ray
from ray.experimental import tqdm_ray
import numpy.matlib
from PyFoam.Basics.DataStructures import Field
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile


def extract_val(case_path: str, time_name: str, x_names: list, cells: list = None):
    """CaseDir,time,cell_indxを指定して変数を読み取る関数

    Args:
        case_path (str): OpenFOAMのcase_path. ex) "./mixerVessel2D"
        time_name (str): OpenFOAMのcase内の時間ディレクトリ名 ex) "0"
        x_names (list): OpenFOAMのcaseの時間ディレクトリ内の変数名 ex) "U"
        cells (list, optional): 出力したいCell番号

    Returns:
        np.array: 一次元配列
        ex) x_names="U"で、cells = [0,2]の時
            (Ux0, Ux2, Uy0, ... , Uz2)
    """
    # return array , init empty
    Xi = np.empty([0, 1])

    for x in x_names:
        # etract val from file
        Xfile = ParsedParameterFile(os.path.join(case_path, time_name, x))
        Xai = np.array(Xfile.content["internalField"])

        # slice with cells
        if cells is not None:
            Xai = Xai[cells]

        # reshape x_dim x 1 matrix
        Xai = Xai.T.reshape([-1, 1])

        # append to Xi
        Xi = np.vstack([Xi, Xai])

    return Xi.T[0]


def cell_distance(case_path: str):
    """各セル間の距離を表すマトリクス

    Args:
        case_path (str):  OpenFOAMのcase_path. ex) "./mixerVessel2D"

    Returns:
        np.array: 対角成分が0の対称行列(cell数,cell数)
        ex) [0.0, 1.0, 0.5]
            [0.5, 0.0, 0.5]
            [1.0, 0.5, 0.0]
    """
    case_path = os.path.join(case_path)
    case = SolutionDirectory(case_path)
    c = ParsedParameterFile(os.path.join(case.name, "0", "C"))
    ce = np.array(c.content["internalField"])
    mat_d = np.zeros((len(ce), len(ce)))
    for i in range(len(ce)):
        mat_d[i] = np.linalg.norm(ce - ce[i], axis=1)
    return mat_d


def localizemat(cell_distance: np.array, lim=0.1):
    """cell間距離に応じて正規分布する影響度。lim以下は影響度0とみなす。
    sigmaの大きさは最大距離の1/10にしてる(仮)。
    将来的にはgaspari&cohn関数がいいはず

    Args:
        cell_distance (np.array): 対角成分が0の対称行列(cell数,cell数)
        lim (float, optional): . Defaults to 0.1.

    Returns:
        np.array: lim～1の影響度マトリクス(cell数,cell数)を返す
    """
    mat_d = cell_distance
    sigma = mat_d.max() / 10
    localizemat = np.exp(-mat_d * mat_d / (2 * sigma**2))
    localizemat[localizemat < lim] = 0.0
    return localizemat


def invR_nonZero(localizemat: np.array, idx: int, obs_indexes: np.array):
    """idx番のcellに対して、観測indexesのinvRと影響0でないかどうかの行列を返す

    Args:
        localizemat (np.array): im～1の影響度マトリクス(cell数,cell数)
        idx (int): 調査したいcell番号
        obs_indexes (np.array): 観測indexes

    Returns:
        taple: invR: 距離行列rの逆行列(観測indexes数,観測indexes数)
               indx_nozero: 影響度0でないかのboolean行列 ex) (True, False, True...)
    """
    obs_indexes = obs_indexes.reshape(1, -1)[0]
    min_index = (idx // len(localizemat)) * len(localizemat)
    max_index = min_index + len(localizemat)
    limmin = obs_indexes < min_index
    limmax = obs_indexes > max_index
    idx = idx % len(localizemat)
    obs_indexes = obs_indexes % len(localizemat)
    invR_diag = localizemat[idx][obs_indexes].T
    invR_diag[limmin] = 0.0
    invR_diag[limmax] = 0.0
    indx_nozero = invR_diag > 0
    invR = np.diag(invR_diag[indx_nozero])
    return invR, indx_nozero


def cell_indies_layer(n_layer: int, cxyz: np.array):
    """外周層のcell indexes を取得する関数
    mixerVesselにしか使えないはず。

    Args:
        n_layer (int): 抽出したい層数
        cxyz (np.array): cellの中心座標

    Returns:
        np.array : 外周層のcell indexes
    """
    c2r = np.sqrt(cxyz[:, 0] ** 2 + cxyz[:, 1] ** 2 + cxyz[:, 2] ** 2)
    c2r = np.round(c2r, 6)
    r_list_rev = np.sort(np.unique(c2r))[::-1]
    r_list_layer = r_list_rev[:n_layer]
    return np.where(np.isin(c2r, r_list_layer))[0]


def pickup_rewite(
    read_file_path: str, write_file_path: str, cells: list, new_val: float = 2.0
):
    """cell list以外を任意の値に変更して、OFの変数ファイルを再作成する関数

    Args:
        read_file_path (str): 読み込み元の変数ファイルパス. ex) "./mixerVessel2D/6/U"
        write_file_path (str): 書き込み先のファイルパス. ex) "./mixerVessel2D/6/Unew"
        cells (list): 変更したくないCell番号リスト
        new_val (float, optional): 変更後の値. Defaults to 2.0.
    """
    # 読み込み
    Xfile = ParsedParameterFile(read_file_path)
    Xc = Xfile.content["internalField"]
    Xa = np.array(Xc)
    # 加工(cells以外をnew_val)
    index = np.ones(len(Xa), dtype=bool)
    index[cells] = False
    Xa[index] = new_val
    # 書き戻し
    Xfile.content["internalField"] = Field(Xa.tolist(), Xc.name)
    Xfile.writeFileAs(write_file_path)


def update_of(x_data: np.ndarray, case_path: str, time_name: str, x_names: list):
    """da後の変数をOFの変数ファイルに書き戻す関数

    Args:
        x_data (np.ndarray): 一次元配列 ex) x_names="U"で、cells = [0,2]の時
                                        (Ux0, Ux2, Uy0, ... , Uz2)
        case_path (str): OpenFOAMのcase_path. ex) "./mixerVessel2D"
        time_name (str): OpenFOAMのcase内の時間ディレクトリ名 ex) "0"
        x_names (list): OpenFOAMのcaseの時間ディレクトリ内の変数名 ex) "U"
    """
    for ofx in x_names:
        # 変数ファイルの読み込み
        Xfile = ParsedParameterFile(os.path.join(case_path, time_name, ofx))
        Xc = Xfile.content["internalField"]
        # internalFieldの値作成
        n_column = 1
        if "vector" in Xc.name:
            n_column = 3
        Xa = x_data[: n_column * len(Xc)].reshape([n_column, -1]).T
        # 次の変数用に今回のデータをスライスで削除
        # x_data = x_data[n_column * len(Xc) :]
        # 書き戻し
        Xfile.content["internalField"] = Field(Xa.tolist(), Xc.name)
        Xfile.writeFile()


def decimal_normalize(floatOrInt, digit: int = 5):
    """OpenFOAMの時間ディレクトリのように数値を丸めたテキストで返すex) 3.000-> 3

    Args:
        floatOrInt (int or float): 変換したい数字
        digit (int, optional): まるめ桁数. Defaults to 5.

    Returns:
        str : まるめた数字 ex) 3, 3.1, 3.01
    """
    text = str(round(floatOrInt, digit))
    while True:
        if ("." in text and text[-1] == "0") or (text[-1] == "."):
            text = text[:-1]
            continue
        break
    return text


def letkf_update(
    xf: np.array,
    H: np.array,
    y0: np.array,
    y_indexes: list,
    lmat: np.array,
    num_cpus: int,
):
    """LETKFによりアンサンブル予報と観測値から解析値(データ同化)を計算する。

    Args:
        xf (np.array): アンサンブル予報マトリクス(アンサンブル数, 状態変数の数)
        H (np.array): 観測演算マトリクス（観測点数, 状態変数の数）
        y0 (np.array): 観測データ(観測点数)
        y_indexes (list): 状態変数に対応する観測インデックスのリスト
        lmat (np.array): cell間距離に応じて正規分布する影響度マトリクス
        num_cpus (int): 並列コア数

    Returns:
        np.array: データ同化後の解析マトリクス(アンサンブル数, 状態変数の数)
    """
    nmem = xf.shape[0]
    dim_x = xf.shape[1]
    xfa = np.mean(xf, axis=0)
    dxf = xf - xfa
    Hxf = H @ xf.T
    dyf = Hxf - H @ xfa.reshape(-1, 1)

    @ray.remote
    def xaj(j, args, bar):
        y_indexes, dyf, nmem, y0, xf, xfa, dxf, lmat, Hxf = args
        invR, nzero = invR_nonZero(lmat, j, y_indexes)
        dyfj = dyf[:, nzero]
        C = dyfj @ invR
        w, v = np.linalg.eig(np.identity(nmem) * (nmem - 1) + C @ dyfj.T)
        w = np.real(w)
        v = np.real(v)
        p_invsq = np.diag(1 / np.sqrt(w))
        p_inv = np.diag(1 / w)
        Wa = v @ p_invsq @ v.T
        Was = v @ p_inv @ v.T

        yHxf_nzero = y0[nzero] - Hxf.mean(axis=1)[nzero]
        xaj = xfa[j] + dxf[:, j] @ (Was @ C @ yHxf_nzero.T + np.sqrt(nmem - 1) * Wa)

        # for progress bar
        bar.update.remote(1)

        return xaj

    # for ray put
    ray.init(num_cpus=num_cpus)
    argset = [y_indexes, dyf, nmem, y0, xf, xfa, dxf, lmat, Hxf]
    argset_ids = ray.put(argset)

    # for progress bar
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    bar = remote_tqdm.remote(total=dim_x)

    # parallel progress
    rayget = ray.get([xaj.remote(j, argset_ids, bar) for j in range(dim_x)])
    ray.shutdown()

    xa = np.array(rayget)
    return xa
