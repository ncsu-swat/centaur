
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_inverse_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    x = np.array([3, 4, 1, 3, 1, 5], dtype=np.int32)
    size = 6
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 1D float32 array with negative values
    x = np.array([-2.5, 0.0, 1.2, -2.5, 3.8, 1.2], dtype=np.float32)
    size = 10
    fill_value = -999.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D int64 array
    x = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int64)
    size = 5
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 2D float64 array
    x = np.array([[-1.0, 2.5], [3.0, -1.0], [0.0, 2.5]], dtype=np.float64)
    size = 8
    fill_value = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 3D int32 array
    x = np.arange(24, dtype=np.int32).reshape(2, 3, 4) % 5
    size = 7
    fill_value = -1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 1D float32 array with size smaller than unique elements
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    size = 3
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 3D float32 array
    x = np.array(
        [[[1.0, -2.0], [3.0, 1.0]], [[-2.0, 4.0], [5.0, 3.0]]],
        dtype=np.float32,
    )
    size = 10
    fill_value = 100.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: 4D int32 array
    x = np.ones((2, 2, 2, 2), dtype=np.int32)
    size = 4
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 1D int64 array with repetitive values
    x = np.array([7, 7, 7, 7, 7], dtype=np.int64)
    size = 2
    fill_value = 42.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 2D float32 array with zero and fractional values
    x = np.array([[0.0, 0.5, 0.25], [0.5, 0.0, 0.75]], dtype=np.float32)
    size = 6
    fill_value = 2.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_inverse_3"] = unique_inverse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_inverse_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_inverse_3'.")


check_valid('jax.numpy.unique_inverse', generated_inputs['jax.numpy.unique_inverse_3'], lib="jax", suffix=3)
