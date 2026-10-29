
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def acosh_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 (values >= 1)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array float32 with values >= 1
    x = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array float64 with values >= 1
    x = np.array([[1.0, 2.5], [3.0, 4.2]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D array float32 including values < 1 (evaluates to NaN)
    x = np.array([0.5, 0.0, -1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array float32
    x = np.linspace(1.0, 50.0, 24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array complex64
    x = np.array([1.0 + 2.0j, -0.5 + 0.5j, 0.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array complex128
    x = np.array(
        [[1.5 - 1.0j, 2.0 + 0.0j], [-2.0 + 3.0j, 0.5 - 0.5j]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array float32
    x = np.ones((2, 2, 2, 2), dtype=np.float32) * 3.0
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: float16 array
    x = np.array([1.0, 1.5, 2.0, 10.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Empty array
    x = np.empty((0, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Array with very large values (testing numerical range)
    x = np.array([1e5, 1e10, 1e20], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.lax.acosh"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.acosh'.")


check_valid('jax.lax.acosh', generated_inputs['jax.lax.acosh'], lib="jax", suffix=0)
