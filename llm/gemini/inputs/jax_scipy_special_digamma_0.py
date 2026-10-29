
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def digamma_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32, positive
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array float32, positive values
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D array float64, negative non-integer values
    x = np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array float32, mix of positive and negative floats
    x = np.array([[-2.3, -0.1], [0.7, 4.2]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array float64, random positive values
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array float32, large values
    x = np.random.uniform(10.0, 100.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array int32 (will be converted/treated as real numbers)
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array float16, small positive values close to zero
    x = np.array([0.001, 0.01, 0.1], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D array float64 with large negative non-integer values
    x = np.array([[-10.5, -20.2], [-30.7, -40.1]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D empty array float32
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 5D array float32
    x = np.ones((1, 2, 1, 3, 1), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.scipy.special.digamma"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.digamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.digamma'.")


check_valid('jax.scipy.special.digamma', generated_inputs['jax.scipy.special.digamma'], lib="jax", suffix=0)
