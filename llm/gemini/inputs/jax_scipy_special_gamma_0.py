
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gamma_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats (float32)
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array of positive and negative non-integers (float64)
    x = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar (0-D array) float32
    x = np.array(4.2, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array of floats (float32)
    x = np.array(
        [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5], [4.0, 5.0, 6.0]], dtype=np.float32
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array of random positive floats (float64)
    x = np.abs(np.random.randn(2, 3, 4)).astype(np.float64) + 0.1
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D complex array (complex64)
    x = np.array(
        [1.0 + 1.0j, 2.0 - 0.5j, -0.5 + 2.0j, 3.0 + 0.0j], dtype=np.complex64
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex array (complex128)
    x = np.array(
        [[1.5 + 0.5j, -1.2 + 0.3j], [2.1 - 1.4j, 0.5 + 0.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array containing special edge cases (zeros, inf, nan)
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D array of small positive values (float32)
    x = np.ones((2, 2, 2, 2), dtype=np.float32) * 0.75
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array with negative integer poles
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D array with larger values to test range
    x = np.array([10.0, 15.0, 20.0, 30.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.scipy.special.gamma"] = gamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.gamma'.")


check_valid('jax.scipy.special.gamma', generated_inputs['jax.scipy.special.gamma'], lib="jax", suffix=0)
