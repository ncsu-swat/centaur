
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polygamma_inputs():
    list_of_inputs = []

    # Input 1: 0D tensor, positive float
    n = np.array(0, dtype=np.int32)
    x = 1.5
    list_of_inputs.append({"n": n, "x": x})

    # Input 2: 1D tensor, positive float
    n = np.array([0, 1, 2], dtype=np.int32)
    x = 2.0
    list_of_inputs.append({"n": n, "x": x})

    # Input 3: 2D tensor, small float
    n = np.array([[1, 0], [2, 3]], dtype=np.int32)
    x = 0.5
    list_of_inputs.append({"n": n, "x": x})

    # Input 4: 1D tensor (int64), larger float
    n = np.array([4, 5], dtype=np.int64)
    x = 10.25
    list_of_inputs.append({"n": n, "x": x})

    # Input 5: 3D tensor, positive float
    n = np.array([[[1, 2], [0, 1]], [[2, 3], [1, 2]]], dtype=np.int32)
    x = 1.0
    list_of_inputs.append({"n": n, "x": x})

    # Input 6: 1D tensor with shape (1,), float
    n = np.array([3], dtype=np.int32)
    x = 4.7
    list_of_inputs.append({"n": n, "x": x})

    # Input 7: 1D tensor with sequential values, float
    n = np.arange(5, dtype=np.int32)
    x = 3.3
    list_of_inputs.append({"n": n, "x": x})

    # Input 8: 2D tensor (int64), float
    n = np.ones((3, 3), dtype=np.int64) * 2
    x = 0.125
    list_of_inputs.append({"n": n, "x": x})

    # Input 9: 1D tensor, float close to 0
    n = np.array([0, 1], dtype=np.int32)
    x = 0.01
    list_of_inputs.append({"n": n, "x": x})

    # Input 10: 4D tensor, float
    n = np.zeros((2, 2, 2, 2), dtype=np.int32)
    x = 5.5
    list_of_inputs.append({"n": n, "x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.polygamma_4"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_4'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_4'], lib="jax", suffix=4)
