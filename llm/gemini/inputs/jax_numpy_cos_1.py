
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cos_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values including positive, negative, and zero
    x = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar array (float64)
    x = np.array(1.2345, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of random standard normal values (float32)
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of integers (int32)
    x = np.arange(-12, 12, dtype=np.int32).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array with float16 dtype
    x = np.linspace(-5.0, 5.0, num=10, dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array of float64 values
    x = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 2)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D complex64 array
    x = np.array([1.0 + 2.0j, -0.5 - 1.5j, 3.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D complex128 array
    x = np.array(
        [[0.0 + 1.0j, -1.0 + 0.0j], [2.0 - 2.0j, -3.0 + 4.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array of int64 values
    x = np.array([-100, -10, 0, 10, 100], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with edge-case values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.cos_1"] = cos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cos_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cos_1'.")


check_valid('jax.numpy.cos', generated_inputs['jax.numpy.cos_1'], lib="jax", suffix=1)
