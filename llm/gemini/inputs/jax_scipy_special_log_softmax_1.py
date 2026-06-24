
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_special_log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis -1
    x = np.random.randn(4, 6).astype(np.float32)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 0
    x = np.random.randn(8, 8).astype(np.float64)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, axis 1
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, axis 2
    x = np.random.randn(2, 2, 5, 5).astype(np.float32)
    input_dict = {"x": x, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with negative values, axis 0
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array with wide range of values, axis -1
    x = np.random.uniform(-50.0, 50.0, size=(2, 3, 3)).astype(np.float64)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, axis 1
    x = np.random.randn(16, 32).astype(np.float32)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32 array, axis -2
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"x": x, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array with very small values, axis 0
    x = np.random.uniform(1e-7, 1e-4, size=(5, 5)).astype(np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.log_softmax_1"] = jax_scipy_special_log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.log_softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.log_softmax_1'.")


check_valid('jax.scipy.special.log_softmax', generated_inputs['jax.scipy.special.log_softmax_1'], lib="jax", suffix=1)
