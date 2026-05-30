
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_elu_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32, alpha is scalar array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    alpha = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of float32, alpha is scalar array
    x = np.random.randn(3, 4).astype(np.float32)
    alpha = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array of float32, alpha is scalar array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    alpha = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 precision
    x = np.random.randn(5, 5).astype(np.float64)
    alpha = np.array(2.0, dtype=np.float64)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float16 precision
    x = np.random.randn(2, 2).astype(np.float16)
    alpha = np.array(1.0, dtype=np.float16)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: alpha is 1D array (broadcasting with trailing dimension)
    x = np.random.randn(4, 3).astype(np.float32)
    alpha = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: alpha has same shape as x
    x = np.random.randn(3, 3).astype(np.float32)
    alpha = np.random.uniform(0.1, 2.0, size=(3, 3)).astype(np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All negative inputs
    x = -np.abs(np.random.randn(10).astype(np.float32))
    alpha = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All positive inputs
    x = np.abs(np.random.randn(10).astype(np.float32))
    alpha = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensor input
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    alpha = np.array(1.2, dtype=np.float32)
    input_dict = {"x": x, "alpha": alpha}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.elu_2"] = jax_nn_elu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.elu_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.elu_2'.")


check_valid('jax.nn.elu', generated_inputs['jax.nn.elu_2'], lib="jax", suffix=2)
