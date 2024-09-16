import numpy as np
import pytest

from foamdapy.variable import esarray


@pytest.fixture
def ensamble_state_array():
    names_to_slices = {"Ux": slice(0, 2), "Uy": slice(2, 4)}
    initial_array = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    return esarray(initial_array, state_dict=names_to_slices)


def test_get_st_value(ensamble_state_array):
    assert np.array_equal(ensamble_state_array.get_st_value("Ux"), np.array([[0, 1], [3, 2]]))


def test_set_st_value(ensamble_state_array):
    ensamble_state_array.set_st_value("Uy", np.array([[4, 5], [6, 7]]))
    assert np.array_equal(ensamble_state_array.get_st_value("Uy"), np.array([[4, 5], [6, 7]]))


def test_invalid_name(ensamble_state_array):
    with pytest.raises(KeyError):
        ensamble_state_array.get_st_value("invalid_name")


def test_esarray_calc():
    # 既存のndarrayをesarrayに変換
    names_to_slices = {"Ux": slice(0, 2), "Uy": slice(2, 4)}
    array1 = esarray(np.array([2, 4, 6, 0]), state_dict=names_to_slices)
    array2 = esarray(np.array([[7, 5], [3, 4], [6, 2], [1, 1]]), state_dict=names_to_slices)

    # 内積を計算\\
    result_dot = np.dot(array1, array2)
    result_diff = array1 - 1

    # 結果の検証
    assert isinstance(result_dot, esarray), "結果がesarrayのインスタンスではありません"
    assert (result_dot == esarray([62, 38])).all(), f"内積の計算結果が正しくありません: {result_dot}"
    assert result_dot.state_dict["Ux"] == slice(
        0, 2), f"state_dict属性が正しく引き継がれていません: {result_dot.state_dict['Ux']}"

    assert isinstance(result_diff, esarray), "結果がesarrayのインスタンスではありません"
    assert (result_diff == esarray([1, 3, 5, -1])).all(), f"内積の計算結果が正しくありません: {result_diff}"
    assert result_diff.state_dict["Uy"] == slice(
        2, 4), f"state_dict属性が正しく引き継がれていません: {result_diff.state_dict['Uy']}"
