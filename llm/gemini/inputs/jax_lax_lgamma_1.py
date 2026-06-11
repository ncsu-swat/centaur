
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lgamma_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive values
    x = np.array([0.5, 1.0, 2.0, 3.5, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array, float64, positive values
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 0D array (scalar), float32
    x = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array, float32, positive and negative non-integer values
    x = np.array([[[-0.5, 0.5], [1.5, -1.5]], [[2.5, -2.5], [3.5, -3.5]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array, float16
    x = np.array([0.1, 1.2, 2.3], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array, float64, small positive values
    x = np.ones((2, 2, 2, 2), dtype=np.float64) * 0.1
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array, float32, large positive values
    x = np.array([[10.0, 20.0], [50.0, 100.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array, float32, negative non-integers specifically to avoid poles
    x = np.array([-0.1, -1.2, -2.7, -3.9], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 3D array, float64, values very close to zero but positive
    x = np.array([[[1e-5, 2e-5], [3e-5, 4e-5]]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array, float32, positive integers as floats
    x = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.lgamma_1"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lgamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lgamma_1'.")


check_valid('jax.lax.lgamma', generated_inputs['jax.lax.lgamma_1'], lib="jax", suffix=1)
