
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def companion_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, minimal length N=2
    a = np.array([2.0, -4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: 1D float32 array, length 4 (matching example)
    a = np.array([1.0, -10.0, 31.0, -30.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: 1D float64 array, length 5 with negative leading coefficient
    a = np.array([-3.5, 2.0, 0.0, -1.5, 4.2], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: 2D batched float32 array, shape (3, 4)
    a = np.array(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, -1.0, 3.0], [5.0, -4.0, 1.0, -2.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: 2D batched float64 array, shape (2, 3)
    a = np.array(
        [[4.0, -1.0, 0.5], [2.5, 3.0, -7.0]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: 3D batched float32 array, shape (2, 2, 4)
    rng = np.random.default_rng(42)
    a = rng.standard_normal((2, 2, 4)).astype(np.float32)
    a[..., 0] += np.sign(a[..., 0]) + 1.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: 1D complex64 array
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: 1D complex128 array, length 4
    a = np.array([2.5 - 1.5j, 0.0 + 3.0j, -1.0 - 1.0j, 5.0 + 0.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: 2D batched complex64 array, shape (2, 3)
    a = np.array(
        [
            [1.0 + 1.0j, -2.0 + 0.0j, 3.0 - 1.0j],
            [-3.0 - 2.0j, 0.5 + 1.5j, 1.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: 1D float64 array with larger degree polynomial (N=8)
    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 11: 4D batched float32 array, shape (1, 2, 3, 3)
    a = rng.standard_normal((1, 2, 3, 3)).astype(np.float32)
    a[..., 0] += 2.0
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.companion"] = companion_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.companion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.companion'.")


check_valid('jax.scipy.linalg.companion', generated_inputs['jax.scipy.linalg.companion'], lib="jax", suffix=0)
