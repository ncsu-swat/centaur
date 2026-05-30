
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def degrees_inputs():
    list_of_inputs = []

    # Input 1: Standard python float (0.0)
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard python float (pi)
    input_dict = {"x": 3.141592653589793}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Standard python float (negative)
    input_dict = {"x": -1.5707963267948966}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: np.float32 positive scalar
    input_dict = {"x": np.float32(1.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: np.float32 negative scalar
    input_dict = {"x": np.float32(-2.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: np.float64 positive scalar
    input_dict = {"x": np.float64(10.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: np.float64 negative scalar
    input_dict = {"x": np.float64(-100.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very large float
    input_dict = {"x": 1e6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very small float
    input_dict = {"x": 1e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Infinity float
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN float
    input_dict = {"x": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.degrees_2"] = degrees_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.degrees_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.degrees_2'.")


check_valid('jax.numpy.degrees', generated_inputs['jax.numpy.degrees_2'], lib="jax", suffix=2)
