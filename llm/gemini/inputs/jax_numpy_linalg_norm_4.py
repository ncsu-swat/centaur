
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_inputs():
    list_of_inputs = []

    # Input 1: 2D array, Frobenius norm, keepdims False
    x = np.array([[-1.5, 2.0, -3.5], [4.0, -5.5, 6.0]], dtype=np.float32)
    input_dict = {"x": x, "ord": "fro", "axis": (0, 1), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, Nuclear norm, keepdims True
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"x": x, "ord": "nuc", "axis": (0, 1), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array (batched matrices), Frobenius norm over axes (1, 2)
    x = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {"x": x, "ord": "fro", "axis": (1, 2), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array (batched matrices), Nuclear norm over axes (1, 2), keepdims True
    x = np.random.randn(2, 4, 5).astype(np.float64)
    input_dict = {"x": x, "ord": "nuc", "axis": (1, 2), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with negative axis indices, Frobenius norm
    x = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict = {"x": x, "ord": "fro", "axis": (-2, -1), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, batched Nuclear norm over axes (-2, -1), keepdims True
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    input_dict = {"x": x, "ord": "nuc", "axis": (-2, -1), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex array, Frobenius norm
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    input_dict = {"x": x, "ord": "fro", "axis": (0, 1), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex array, Nuclear norm, keepdims True
    x = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(
        np.complex128
    )
    input_dict = {"x": x, "ord": "nuc", "axis": (0, 1), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, Frobenius norm across non-consecutive axes (0, 2)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"x": x, "ord": "fro", "axis": (0, 2), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, Frobenius norm across reversed axes (-1, -2)
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {"x": x, "ord": "fro", "axis": (-1, -2), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array, Nuclear norm across axes (0, 1)
    x = np.random.randn(3, 3, 2).astype(np.float32)
    input_dict = {"x": x, "ord": "nuc", "axis": (0, 1), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.norm_4"] = norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_4'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_4'], lib="jax", suffix=4)
