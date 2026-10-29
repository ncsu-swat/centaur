
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def acosh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with values >= 1
    x = np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array with values >= 1
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar (0D array)
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float32 array
    x = np.ones((2, 3, 4), dtype=np.float32) * 3.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: Complex128 array (acosh is well-defined over complex plane)
    x = np.array([0.5 + 1.0j, -2.0 + 0.5j, 1.0 - 1.0j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Integer array (values >= 1)
    x = np.array([1, 2, 5, 10, 100], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Float array including values < 1 (produces NaN in real domain, standard behaviour)
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Float16 array
    x = np.array([[1.0, 1.2], [1.8, 2.5]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D float32 array
    x = np.linspace(1.0, 10.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array containing infinity
    x = np.array([1.0, 10.0, np.inf], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.acosh"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.acosh'.")


check_valid('jax.numpy.acosh', generated_inputs['jax.numpy.acosh'], lib="jax", suffix=0)
