
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def block_diag_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 square matrix
    arrs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 2: 2D float64 matrix with mixed signs
    arrs = np.array(
        [[-1.5, 2.5, 0.0], [3.0, -4.5, 1.2], [0.0, -2.1, 5.5]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 3: 2D float32 wide rectangular matrix
    arrs = np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 4: 2D int32 matrix with negative numbers
    arrs = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 5: 2D complex64 matrix
    arrs = np.array(
        [[1.0 + 2.0j, -3.0j], [4.0 - 1.0j, 0.5 + 0.5j]], dtype=np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 6: 1D float32 vector
    arrs = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 7: 2D boolean matrix
    arrs = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 8: 2D float64 tall rectangular matrix
    arrs = np.arange(10, dtype=np.float64).reshape(5, 2) * 0.5
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 9: 2D int64 matrix (1x4 row vector)
    arrs = np.array([[100, -200, 300, -400]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 10: 1D int32 vector
    arrs = np.array([-5, 0, 5, 10, -15], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 11: 0D scalar array (float32)
    arrs = np.array(42.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 12: 2D complex128 matrix
    arrs = np.array([[-1.0 + 1.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.block_diag"] = block_diag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.block_diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.block_diag'.")


check_valid('jax.scipy.linalg.block_diag', generated_inputs['jax.scipy.linalg.block_diag'], lib="jax", suffix=0)
