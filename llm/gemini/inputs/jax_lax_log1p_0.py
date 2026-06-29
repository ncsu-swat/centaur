
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log1p_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values, accuracy=None
    x = np.array([0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array with mixed positive and negative values (>-1), accuracy=None
    x = np.array([[-0.5, 0.0], [0.5, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array with very small values (near zero), accuracy=None
    x = np.array([[[1e-15, 1e-10], [1e-8, 1e-5]]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar float32, accuracy=None
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float64 array with large values, accuracy=None
    x = np.ones((2, 2, 2, 2), dtype=np.float64) * 1000.0
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex64 array, accuracy=None
    x = np.array([0.5 + 0.5j, -0.2 + 0.1j], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex128 array, accuracy=None
    x = np.array([[1.0 + 2.0j, -0.5 - 0.5j], [0.0 + 0.0j, 10.0 + 0.0j]], dtype=np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array containing all zeros, accuracy=None
    x = np.zeros((2, 3, 4), dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array with values close to -1, accuracy=None
    x = np.array([[-0.9, -0.99], [-0.999, -0.9999]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array with extremely small values, accuracy=None
    x = np.array([1e-30, 1e-20, 1e-16], dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.log1p'.")


check_valid('jax.lax.log1p', generated_inputs['jax.lax.log1p'], lib="jax", suffix=0)
