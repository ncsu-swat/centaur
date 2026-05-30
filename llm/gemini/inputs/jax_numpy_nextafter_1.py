
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nextafter_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 1.0, 4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays
    x = np.array([[1.0, -1.0], [0.0, 2.0]], dtype=np.float32)
    y = np.array([[2.0, -2.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 precision, 1D
    x = np.array([1e-10, 1e10], dtype=np.float64)
    y = np.array([1e-9, 1e11], dtype=np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float16 precision
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float16)
    y = np.array([0.0, -1.0, 2.0], dtype=np.float16)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting (x is 2D, y is 1D)
    x = np.random.randn(3, 3).astype(np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting (x is 1D shape (1,), y is 2D)
    x = np.array([0.0], dtype=np.float32)
    y = np.random.randn(2, 2).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Special values (infinity, zero)
    x = np.array([0.0, -0.0, np.inf, -np.inf], dtype=np.float32)
    y = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    x = np.array([1e30, -1e30], dtype=np.float32)
    y = np.array([np.inf, -np.inf], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 arrays
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nextafter_1"] = nextafter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nextafter_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nextafter_1'.")


check_valid('jax.numpy.nextafter', generated_inputs['jax.numpy.nextafter_1'], lib="jax", suffix=1)
