
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fill_diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix with 1D val matching diagonal length
    a = np.zeros((3, 3), dtype=np.float32)
    val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 2: 2D non-square matrix (more cols than rows) with 0D scalar tensor val
    a = np.ones((3, 5), dtype=np.int32)
    val = np.array(-5, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 3: 2D non-square matrix (more rows than cols) with 1D val shorter than diagonal
    a = np.zeros((6, 2), dtype=np.float64)
    val = np.array([4.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 4: 3D cube array (all dims equal) with 1D val
    a = np.zeros((2, 2, 2), dtype=np.float32)
    val = np.array([10.0, 20.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 5: 4D hypercube array (all dims equal) with 1D val
    a = np.zeros((3, 3, 3, 3), dtype=np.int64)
    val = np.array([-1, -2, -3], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 6: 2D square complex128 array
    a = np.zeros((4, 4), dtype=np.complex128)
    val = np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex128)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 7: Boolean matrix with boolean scalar tensor
    a = np.zeros((4, 4), dtype=bool)
    val = np.array(True, dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 8: 2D square matrix with 1D val longer than diagonal (gets truncated)
    a = np.full((3, 3), -10.0, dtype=np.float32)
    val = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 9: 3D cube array with 2D val (gets flattened)
    a = np.zeros((3, 3, 3), dtype=np.float64)
    val = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 10: 2D non-square matrix with int32 dtype
    a = np.zeros((4, 7), dtype=np.int32)
    val = np.array([10, 20, 30, 40], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    # Input 11: 1x1 2D matrix
    a = np.array([[7.0]], dtype=np.float32)
    val = np.array([42.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "val": val, "wrap": False, "inplace": False})
    )

    return list_of_inputs


generated_inputs["jax.numpy.fill_diagonal_1"] = fill_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fill_diagonal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fill_diagonal_1'.")


check_valid('jax.numpy.fill_diagonal', generated_inputs['jax.numpy.fill_diagonal_1'], lib="jax", suffix=1)
