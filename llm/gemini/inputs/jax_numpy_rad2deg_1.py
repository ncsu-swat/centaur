
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rad2deg_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32
    x = np.array([0.0, np.pi / 4, np.pi / 2, np.pi], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with negative values
    x = np.array([-np.pi, -np.pi / 2, -np.pi / 3, -np.pi / 6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar as 0D tensor
    x = np.array(np.pi, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array of float32
    x = np.random.uniform(-np.pi, np.pi, size=(3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of float64
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(2, 4, 3)).astype(
        np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Integer tensor
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Float16 tensor
    x = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D tensor
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with special values (inf, -inf, nan)
    x = np.array([-np.inf, -1.0, 0.0, np.nan, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array of large angles
    x = np.array([10 * np.pi, -100 * np.pi, 1e5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Int64 tensor with 2D shape
    x = np.arange(-6, 6, dtype=np.int64).reshape((3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.rad2deg_1"] = rad2deg_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rad2deg_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rad2deg_1'.")


check_valid('jax.numpy.rad2deg', generated_inputs['jax.numpy.rad2deg_1'], lib="jax", suffix=1)
