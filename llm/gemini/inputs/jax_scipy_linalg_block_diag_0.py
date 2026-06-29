
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def block_diag_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 square matrix
    arrs = np.ones((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 2: 2D float64 rectangular matrix
    arrs = np.random.randn(2, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 3: 1D float32 array
    arrs = np.random.randn(5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 4: 2D int32 square matrix
    arrs = np.arange(16, dtype=np.int32).reshape(4, 4)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 5: 2D float16 matrix
    arrs = np.random.randn(2, 2).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 6: 2D float32 matrix with negative values
    arrs = -np.ones((3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 7: 2D complex64 matrix
    arrs = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 8: 2D boolean matrix
    arrs = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 9: 1D int16 array
    arrs = np.arange(10, dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"arrs": arrs}))

    # Input 10: Larger 2D float32 matrix
    arrs = np.random.randn(10, 10).astype(np.float32)
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
