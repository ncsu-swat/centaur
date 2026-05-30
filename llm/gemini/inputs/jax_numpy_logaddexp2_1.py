
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logaddexp2_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, positive values
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 2D arrays, float32, positive and negative values
    x1 = np.array([[1.5, -2.3], [0.0, 4.1]], dtype=np.float32)
    x2 = np.array([[-0.5, 2.3], [1.1, -3.2]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: float64 precision for high accuracy
    x1 = np.random.uniform(-10.0, 10.0, size=(3, 3)).astype(np.float64)
    x2 = np.random.uniform(-10.0, 10.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Broadcasting (1D to 2D)
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x2 = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: Broadcasting (scalar-like 1x1 to 3x3)
    x1 = np.random.randn(3, 3).astype(np.float32)
    x2 = np.array([[2.0]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: Large positive values to test overflow avoidance
    x1 = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    x2 = np.array([1001.0, 1999.0, 3000.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: Large negative values to test underflow avoidance
    x1 = np.array([-1000.0, -2000.0, -3000.0], dtype=np.float32)
    x2 = np.array([-1001.0, -1999.0, -3000.0], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: Float16 arrays
    x1 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    x2 = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float16)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: 3D arrays with mixed signs
    x1 = np.random.uniform(-50.0, 50.0, size=(2, 2, 2)).astype(np.float32)
    x2 = np.random.uniform(-50.0, 50.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: Broadcasting with incompatible-looking but valid shapes (e.g., (3, 1) and (1, 4))
    x1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    x2 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 11: Higher dimensional array (4D)
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x2 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.logaddexp2_1"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp2_1'.")


check_valid('jax.numpy.logaddexp2', generated_inputs['jax.numpy.logaddexp2_1'], lib="jax", suffix=1)
