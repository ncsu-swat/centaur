
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import scipy.linalg
import copy

class SafeTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.concatenate([np.atleast_1d(x).ravel() for x in self])

def lu_solve_inputs():
    list_of_inputs = []

    # Input 1: 2x2 float32, b 1D, trans=0, default flags
    a = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    lu, piv = scipy.linalg.lu_factor(a)
    b = np.array([3.0, 4.0], dtype=np.float32)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 0,
        "overwrite_b": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 float64, b 2D, trans=1, modified flags
    a = np.random.randn(3, 3).astype(np.float64)
    lu, piv = scipy.linalg.lu_factor(a)
    b = np.random.randn(3, 2).astype(np.float64)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 1,
        "overwrite_b": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4x4 complex64, b 1D, trans=2
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    lu, piv = scipy.linalg.lu_factor(a)
    b = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 2,
        "overwrite_b": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 5x5 complex128, b 2D, trans=0
    a = (np.random.randn(5, 5) + 1j * np.random.randn(5, 5)).astype(np.complex128)
    lu, piv = scipy.linalg.lu_factor(a)
    b = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex128)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 0,
        "overwrite_b": True,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched (2, 3, 3) float32, b is (2, 3), trans=0
    lus, pivs = [], []
    for _ in range(2):
        a_sub = np.random.randn(3, 3).astype(np.float32)
        lu_sub, piv_sub = scipy.linalg.lu_factor(a_sub)
        lus.append(lu_sub)
        pivs.append(piv_sub)
    lu = np.stack(lus)
    piv = np.stack(pivs)
    b = np.random.randn(2, 3).astype(np.float32)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 0,
        "overwrite_b": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched (3, 4, 4) float64, b is (3, 4, 2), trans=1
    lus, pivs = [], []
    for _ in range(3):
        a_sub = np.random.randn(4, 4).astype(np.float64)
        lu_sub, piv_sub = scipy.linalg.lu_factor(a_sub)
        lus.append(lu_sub)
        pivs.append(piv_sub)
    lu = np.stack(lus)
    piv = np.stack(pivs)
    b = np.random.randn(3, 4, 2).astype(np.float64)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 1,
        "overwrite_b": False,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 6x6 float32, b 1D, trans=1
    a = np.random.randn(6, 6).astype(np.float32)
    lu, piv = scipy.linalg.lu_factor(a)
    b = np.random.randn(6).astype(np.float32)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 1,
        "overwrite_b": True,
        "check_finite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 10x10 float64, b 2D, trans=0
    a = np.random.randn(10, 10).astype(np.float64)
    lu, piv = scipy.linalg.lu_factor(a)
    b = np.random.randn(10, 5).astype(np.float64)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 0,
        "overwrite_b": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Batched (2, 2, 5, 5) float32, b is (2, 2, 5), trans=0
    lus, pivs = [], []
    for _ in range(4):
        a_sub = np.random.randn(5, 5).astype(np.float32)
        lu_sub, piv_sub = scipy.linalg.lu_factor(a_sub)
        lus.append(lu_sub)
        pivs.append(piv_sub)
    lu = np.stack(lus).reshape(2, 2, 5, 5)
    piv = np.stack(pivs).reshape(2, 2, 5)
    b = np.random.randn(2, 2, 5).astype(np.float32)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 0,
        "overwrite_b": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 8x8 float32, b 2D, trans=2
    a = np.random.randn(8, 8).astype(np.float32)
    lu, piv = scipy.linalg.lu_factor(a)
    b = np.random.randn(8, 1).astype(np.float32)
    input_dict = {
        "lu_and_piv": SafeTuple((lu, piv)),
        "b": b,
        "trans": 2,
        "overwrite_b": False,
        "check_finite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.lu_solve"] = lu_solve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.lu_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.lu_solve'.")


check_valid('jax.scipy.linalg.lu_solve', generated_inputs['jax.scipy.linalg.lu_solve'], lib="jax", suffix=0)
