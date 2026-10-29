
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vander_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float array with N equal to length, increasing=False
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    N = int(4)
    increasing = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 2: 1D int array with increasing=True
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    N = int(4)
    increasing = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 3: Float64 array with N smaller than len(x)
    x = np.array([-2.5, -0.5, 0.0, 1.5, 3.0], dtype=np.float64)
    N = int(3)
    increasing = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 4: Float32 array with N larger than len(x)
    x = np.array([0.5, 2.0], dtype=np.float32)
    N = int(5)
    increasing = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 5: N = 1 (constant column)
    x = np.array([10.0, -20.0, 30.0], dtype=np.float32)
    N = int(1)
    increasing = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 6: N = 0 (empty second dimension)
    x = np.array([1, 2, 3], dtype=np.int64)
    N = int(0)
    increasing = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 7: Empty 1D input array
    x = np.array([], dtype=np.float32)
    N = int(3)
    increasing = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 8: Complex-valued 1D array
    x = np.array([1.0 + 2.0j, -0.5 + 1.0j, 3.0 - 1.0j], dtype=np.complex64)
    N = int(3)
    increasing = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 9: Single-element array
    x = np.array([5.0], dtype=np.float64)
    N = int(4)
    increasing = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 10: Negative integers
    x = np.array([-5, -3, -1], dtype=np.int64)
    N = int(2)
    increasing = True
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    # Input 11: Larger array of floats
    x = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    N = int(6)
    increasing = False
    list_of_inputs.append(
        copy.deepcopy({"x": x, "N": N, "increasing": increasing})
    )

    return list_of_inputs


generated_inputs["jax.numpy.vander"] = vander_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vander' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vander'.")


check_valid('jax.numpy.vander', generated_inputs['jax.numpy.vander'], lib="jax", suffix=0)
