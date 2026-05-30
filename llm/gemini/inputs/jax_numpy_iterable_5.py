
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iterable_inputs():
    list_of_inputs = []

    # Input 1: positive float
    list_of_inputs.append(copy.deepcopy({"y": 1.0}))

    # Input 2: zero float
    list_of_inputs.append(copy.deepcopy({"y": 0.0}))

    # Input 3: negative float
    list_of_inputs.append(copy.deepcopy({"y": -5.5}))

    # Input 4: positive infinity
    list_of_inputs.append(copy.deepcopy({"y": float('inf')}))

    # Input 5: negative infinity
    list_of_inputs.append(copy.deepcopy({"y": float('-inf')}))

    # Input 6: NaN
    list_of_inputs.append(copy.deepcopy({"y": float('nan')}))

    # Input 7: np.float32
    list_of_inputs.append(copy.deepcopy({"y": np.float32(3.14)}))

    # Input 8: np.float64
    list_of_inputs.append(copy.deepcopy({"y": np.float64(-123.456)}))

    # Input 9: scientific notation small float
    list_of_inputs.append(copy.deepcopy({"y": 1e-10}))

    # Input 10: scientific notation large float
    list_of_inputs.append(copy.deepcopy({"y": 1e10}))

    # Input 11: np.float16
    list_of_inputs.append(copy.deepcopy({"y": np.float16(0.001)}))

    return list_of_inputs

generated_inputs["jax.numpy.iterable_5"] = iterable_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iterable_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iterable_5'.")


check_valid('jax.numpy.iterable', generated_inputs['jax.numpy.iterable_5'], lib="jax", suffix=5)
