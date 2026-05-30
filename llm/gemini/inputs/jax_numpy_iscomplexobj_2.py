
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"x": 1.0})

    # Input 2: Negative float
    list_of_inputs.append({"x": -5.5})

    # Input 3: Zero float
    list_of_inputs.append({"x": 0.0})

    # Input 4: Large float
    list_of_inputs.append({"x": 1e10})

    # Input 5: Small float (close to zero)
    list_of_inputs.append({"x": 1e-10})

    # Input 6: Positive infinity
    list_of_inputs.append({"x": float('inf')})

    # Input 7: Negative infinity
    list_of_inputs.append({"x": float('-inf')})

    # Input 8: NaN (Not a Number)
    list_of_inputs.append({"x": float('nan')})

    # Input 9: numpy float32 scalar
    list_of_inputs.append({"x": float(np.float32(3.14))})

    # Input 10: numpy float64 scalar
    list_of_inputs.append({"x": float(np.float64(-0.001))})

    # Input 11: Another standard float
    list_of_inputs.append({"x": 42.42})

    return list_of_inputs

generated_inputs["jax.numpy.iscomplexobj_2"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_2'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_2'], lib="jax", suffix=2)
