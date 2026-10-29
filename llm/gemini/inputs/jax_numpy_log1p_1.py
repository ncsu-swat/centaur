
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log1p_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with small positive values
    x = np.array([1e-4, 1e-6, 2e-10], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float64 array with standard positive values
    x = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float32 array with values between -1 and 1
    x = np.array([[-0.5, -0.2], [0.3, 0.8]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar float32 array
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 array
    x = np.random.uniform(low=0.0, high=10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D integer array
    x = np.array([0, 1, 2, 3, 4, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D int64 array
    x = np.array([[0, 5], [10, 15]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: float16 array
    x = np.array([-0.9, -0.1, 0.0, 0.5, 1.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Complex array
    x = np.array([1.0 + 2.0j, -0.5 + 0.5j, 0.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D float64 array
    x = np.random.uniform(low=-0.99, high=5.0, size=(2, 2, 3, 3)).astype(
        np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.log1p_1"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log1p_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log1p_1'.")


check_valid('jax.numpy.log1p', generated_inputs['jax.numpy.log1p_1'], lib="jax", suffix=1)
