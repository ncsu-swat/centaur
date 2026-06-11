
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lgamma_inputs():
    list_of_inputs = []

    # Input 1: float32 scalar
    input_dict = {"x": np.float32(1.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 scalar
    input_dict = {"x": np.float64(2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 array
    input_dict = {"x": np.array([0.5, 1.5, 2.5], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array
    input_dict = {"x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    input_dict = {"x": np.array([[[1.0, 1.5], [2.0, 2.5]], [[3.0, 3.5], [4.0, 4.5]]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64 array
    input_dict = {"x": np.ones((1, 2, 1, 3), dtype=np.float64) * 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D float32 array
    input_dict = {"x": np.array(3.5, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array with valid negative values
    input_dict = {"x": np.array([-0.5, -1.5, 0.1], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D float32 array
    input_dict = {"x": np.random.uniform(0.1, 10.0, size=(5, 5)).astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array
    input_dict = {"x": np.array([5.0, 10.0, 100.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.lgamma_4"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lgamma_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lgamma_4'.")


check_valid('jax.lax.lgamma', generated_inputs['jax.lax.lgamma_4'], lib="jax", suffix=4)
