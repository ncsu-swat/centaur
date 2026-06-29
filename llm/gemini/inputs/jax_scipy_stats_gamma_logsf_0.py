
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gamma_logsf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, basic positive values
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "a": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32, positive shift
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "a": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        "loc": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        "scale": np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D arrays, float64, negative loc shift
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.float64) * 5.0,
        "a": np.ones((2, 2, 2), dtype=np.float64) * 3.0,
        "loc": np.ones((2, 2, 2), dtype=np.float64) * -1.0,
        "scale": np.ones((2, 2, 2), dtype=np.float64) * 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalars wrapped in arrays)
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "a": np.array(4.0, dtype=np.float32),
        "loc": np.array(1.0, dtype=np.float32),
        "scale": np.array(0.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting shapes
    input_dict = {
        "x": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),  # Shape (1, 3)
        "a": np.array([[2.0], [3.0]], dtype=np.float32),      # Shape (2, 1)
        "loc": np.array([0.0], dtype=np.float32),             # Shape (1,)
        "scale": np.array([[1.0]], dtype=np.float32)          # Shape (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x is smaller than loc (sf evaluates to 1, logsf to 0)
    input_dict = {
        "x": np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        "a": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large x values, float64
    input_dict = {
        "x": np.array([100.0, 200.0], dtype=np.float64),
        "a": np.array([2.0, 2.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger dimensions (4D arrays), float32, random valid values
    input_dict = {
        "x": np.random.uniform(5.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32),
        "a": np.random.uniform(1.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32),
        "loc": np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32),
        "scale": np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive parameters (shape and scale)
    input_dict = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "a": np.array([0.01, 0.05, 0.1], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.1, 0.1, 0.1], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed dimension arrays with float64
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "a": np.array([1.5, 2.5], dtype=np.float64),
        "loc": np.array([0.1, 0.2], dtype=np.float64),
        "scale": np.array([1.0, 2.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gamma.logsf"] = gamma_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.logsf'.")


check_valid('jax.scipy.stats.gamma.logsf', generated_inputs['jax.scipy.stats.gamma.logsf'], lib="jax", suffix=0)
