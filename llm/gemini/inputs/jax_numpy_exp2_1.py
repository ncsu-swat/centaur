
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def exp2_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([2.0, -4.0, 3.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D int32 array
    x = np.array([0, 1, 2, 3, 4, -1, -2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array float64
    x = np.array(5.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D float64 matrix with random values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 tensor
    x = np.linspace(-3.0, 3.0, num=24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D float16 tensor
    x = np.ones((2, 2, 3, 3), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D int64 array
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex array (complex64)
    x = np.array([1.0 + 2.0j, -1.0 - 0.5j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array with special values (inf, -inf, nan, zero)
    x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.exp2_1"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp2_1'.")


check_valid('jax.numpy.exp2', generated_inputs['jax.numpy.exp2_1'], lib="jax", suffix=1)
