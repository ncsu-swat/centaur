
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"x": 3.14}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float
    input_dict = {"x": -2.718}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero float
    input_dict = {"x": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large float
    input_dict = {"x": 1.0e15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small float (scientific notation)
    input_dict = {"x": 1.0e-15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Positive infinity float
    input_dict = {"x": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative infinity float
    input_dict = {"x": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: NaN float
    input_dict = {"x": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Numpy float32 scalar
    input_dict = {"x": np.float32(12.34)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Numpy float64 scalar
    input_dict = {"x": np.float64(-56.78)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Numpy float16 scalar
    input_dict = {"x": np.float16(0.001)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isrealobj_3"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_3'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_3'], lib="jax", suffix=3)
