import numpy as np
from scipy.sparse import coo_matrix

from foamdapy.tools import decimal_normalize
from foamdapy.tools import letkf_update
from foamdapy.tools import createRdiag_from_xf


def test_createRdiag_from_xf():
    np.random.seed(0)
    num_ensenble = 20
    x1 = (-6, 8)
    x2 = (0, 8)
    xf = np.stack(
        [
            np.array([x1[0]] * num_ensenble),
            np.array([x1[0] + x1[1]] * num_ensenble),
            np.array([x1[1]] * num_ensenble),
            np.array([x2[0]] * num_ensenble),
            np.array([x2[0] + x2[1]] * num_ensenble),
            np.array([x2[1]] * num_ensenble),
        ],
        axis=1,
    )
    n_cells = 3  # 3セル x 2変数
    obs_indexes = [0, 4]  # 1変数目:0,1,2、　2変数目:3,4,5
    res = createRdiag_from_xf(xf, n_cells, obs_indexes)
    assert res[0] == ((x1[1] - x1[0]) * 0.01 / 2.0) ** 2
    assert res[1] == ((x2[1] - x2[0]) * 0.01 / 2.0) ** 2


def test_decimal_normalize():
    assert decimal_normalize(3.00001) == "3.00001"
    assert decimal_normalize(3.000001) == "3"


def test_letkf_update():
    np.random.seed(0)
    num_ensenble = 40
    num_obs = 2

    xf = np.stack(
        [
            np.random.normal(0.0, 1, num_ensenble),
            np.random.normal(1.0, 2, num_ensenble),
            np.random.normal(2.0, 1, num_ensenble),
        ],
        axis=1,
    )
    # H = np.array([[0, 1, 0], [0, 0, 1]])
    Hcoo = coo_matrix(([1, 1], ([0, 1], [1, 2])))
    Hlil = Hcoo.tolil()
    t0 = xf * 2.0
    y0 = t0[:, -num_obs:].mean(axis=0)
    y_indexes = np.array([1, 2])
    R_diag = createRdiag_from_xf(xf, 3, y_indexes)
    lmat = np.full((3, 3), 1.0)
    num_cpu = 1
    xa_loop = letkf_update(xf, Hlil, y0, R_diag, y_indexes, lmat, num_cpu)
    xa1 = xa_loop.copy()
    for i in range(10):
        xa_loop = letkf_update(xa_loop, Hlil, y0, R_diag, y_indexes, lmat, num_cpu)

    # test of converges to observed value
    assert (np.round(xa_loop.mean(axis=0)[-num_obs:], 3) == np.round(y0, 3)).all()

    # parallel test
    num_cpu = 2
    xa2 = letkf_update(xf, Hlil, y0, R_diag, y_indexes, lmat, num_cpu)
    assert (np.round(xa1.mean(axis=0), 9) == np.round(xa2.mean(axis=0), 9)).all()
