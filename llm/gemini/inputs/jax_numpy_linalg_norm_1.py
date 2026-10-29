
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, ord=1, axis=0, keepdims=False
    x = np.array([3.0, -4.0, 12.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 1, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, ord=2 (Euclidean), axis=-1, keepdims=True
    x = np.array([-1.5, 2.0, -3.5, 4.0], dtype=np.float64)
    input_dict = {"x": x, "ord": 2, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, ord=0 (count non-zero), axis=0, keepdims=False
    x = np.array([0.0, 1.0, -2.0, 0.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 0, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, ord=3 (p-norm), axis=1, keepdims=False
    x = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    input_dict = {"x": x, "ord": 3, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, ord=1 (L1 vector norm along axis 0), axis=0, keepdims=True
    x = np.arange(12, dtype=np.float64).reshape(3, 4) - 5.0
    input_dict = {"x": x, "ord": 1, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, ord=2 (L2 vector norm along axis -2), axis=-2, keepdims=False
    x = np.random.randn(4, 5).astype(np.float32)
    input_dict = {"x": x, "ord": 2, "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, ord=1, axis=2, keepdims=True
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "ord": 1, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, ord=4, axis=0, keepdims=False
    x = np.random.randn(3, 2, 5).astype(np.float64)
    input_dict = {"x": x, "ord": 4, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, ord=2, axis=1, keepdims=True
    x = np.random.randn(2, 3, 2, 4).astype(np.float32)
    input_dict = {"x": x, "ord": 2, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, ord=0, axis=-1, keepdims=False
    x = np.array([[[[0.0, 1.0], [2.0, 0.0]]]], dtype=np.float32)
    input_dict = {"x": x, "ord": 0, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array with complex dtype, ord=2, axis=0, keepdims=False
    x = np.array(
        [[1.0 + 2.0j, -3.0 + 4.0j], [5.0 - 1.0j, 0.0 + 0.0j]],
        dtype=np.complex64,
    )
    input_dict = {"x": x, "ord": 2, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 1D array, ord=5, axis=0, keepdims=True
    x = np.array([-2.0, 3.0, -1.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "ord": 5, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.norm_1"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_1'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_1'], lib="jax", suffix=1)
