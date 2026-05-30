
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atan2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, positive values
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 1D float32, mixed positive, negative, and zero values
    x1 = np.array([-1.0, 0.0, 1.0, -2.0], dtype=np.float32)
    x2 = np.array([1.0, -1.0, 0.0, -2.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: 2D float64, random values
    x1 = np.random.uniform(-10, 10, size=(3, 3)).astype(np.float64)
    x2 = np.random.uniform(-10, 10, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: 3D float32, random values
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: 1D int32
    x1 = np.array([1, -2, 3], dtype=np.int32)
    x2 = np.array([-4, 5, -6], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: 2D int64
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    x2 = np.array([[-1, -2], [-3, -4]], dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: 0D arrays (scalars as arrays)
    x1 = np.array(1.5, dtype=np.float32)
    x2 = np.array(-2.5, dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: Broadcasting shapes (2D and 1D)
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(4).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Large values float64
    x1 = np.array([1e10, -1e10, 0.0], dtype=np.float64)
    x2 = np.array([-1e10, 1e10, 1e10], dtype=np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: 4D float32
    x1 = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 11: Very small values close to zero
    x1 = np.array([1e-10, -1e-10], dtype=np.float32)
    x2 = np.array([-1e-10, 1e-10], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.atan2"] = atan2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atan2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atan2'.")


check_valid('jax.numpy.atan2', generated_inputs['jax.numpy.atan2'], lib="jax", suffix=0)
