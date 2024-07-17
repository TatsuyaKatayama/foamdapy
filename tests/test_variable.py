from foamdapy.variable import esarray
import numpy as np

import pytest

@pytest.fixture
def ensamble_state_array():
    names_to_slices = {"Ux": slice(0, 2), "Uy": slice(2, 4)}
    initial_array = np.array([[0,1,2,3],[3,2,1,0]])
    return esarray(initial_array, state_dict=names_to_slices)

def test_get_st_value(ensamble_state_array):
    assert np.array_equal(ensamble_state_array.get_st_value("Ux"), np.array([[0, 1],[3,2]]))

def test_set_st_value(ensamble_state_array):
    ensamble_state_array.set_st_value("Uy", np.array([[4, 5],[6,7]]))
    assert np.array_equal(ensamble_state_array.get_st_value("Uy"), np.array([[4, 5],[6,7]]))

def test_invalid_name(ensamble_state_array):
    with pytest.raises(KeyError):
        ensamble_state_array.get_st_value("invalid_name")
