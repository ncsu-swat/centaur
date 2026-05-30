
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def float_power_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([3.0, 2.0, 0.5], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 arrays
    x = np.array([3, 1, -5], dtype=np.int32)
    y = np.array([2, 4, -1], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcast compatible, 2D and 1D
    x = np.array([[2, -4, 1], [-1, 2, 3]], dtype=np.float32)
    y = np.array([-2, 1, 4], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar arrays (0D)
    x = np.array(5.0, dtype=np.float32)
    y = np.array(-2.0, dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 high precision
    x = np.random.uniform(0.1, 10.0, size=(5, 5)).astype(np.float64)
    y = np.random.uniform(-2.0, 2.0, size=(5, 5)).astype(np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors, same shape
    x = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    y = np.random.uniform(0.0, 3.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting with 1s in dimensions
    x = np.random.uniform(1.0, 5.0, size=(1, 3, 1)).astype(np.float32)
    y = np.random.uniform(0.0, 3.0, size=(4, 1, 5)).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Exponents are negative integers, base is float
    x = np.array([0.5, 2.0, 10.0], dtype=np.float32)
    y = np.array([-3, -2, -1], dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large exponents with float32
    x = np.array([1.001, 0.999], dtype=np.float32)
    y = np.array([1000, 1000], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed types (float16 and float32)
    x = np.array([2.0, 4.0, 16.0], dtype=np.float16)
    y = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.float_power_1"] = float_power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.float_power_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.float_power_1'.")


check_valid('jax.numpy.float_power', generated_inputs['jax.numpy.float_power_1'], lib="jax", suffix=1)
