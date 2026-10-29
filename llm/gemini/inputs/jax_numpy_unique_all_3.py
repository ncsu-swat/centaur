
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_all_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with duplicates
    x = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 5.0], dtype=np.float32)
    size = 4
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float32 array with negative values
    x = np.array([[-1.5, 2.0], [2.0, -3.0], [-1.5, 0.5]], dtype=np.float32)
    size = 6
    fill_value = -1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 1D float64 array with size smaller than unique count
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    size = 3
    fill_value = -999.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 3D float32 array
    x = np.ones((2, 2, 3), dtype=np.float32)
    x[0, 1, :] = [2.0, 3.0, 4.0]
    size = 8
    fill_value = 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 2D float64 array with size larger than unique count
    x = np.array([[10.0, 20.0], [20.0, 10.0]], dtype=np.float64)
    size = 5
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 1D int32 array
    x = np.array([4, 2, 2, 8, 4, -1, -1], dtype=np.int32)
    size = 6
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 4D float32 array
    x = np.arange(16, dtype=np.float32).reshape((2, 2, 2, 2))
    size = 16
    fill_value = -0.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: 1D array with identical elements
    x = np.full((10,), 7.5, dtype=np.float32)
    size = 3
    fill_value = 100.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 2D int64 array
    x = np.array([[-5, -10], [-5, 0], [10, -10]], dtype=np.int64)
    size = 5
    fill_value = -1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 3D float64 array with random values
    x = np.array(
        [[[0.1, 0.2], [0.1, 0.4]], [[-0.5, 0.4], [0.2, -0.5]]],
        dtype=np.float64,
    )
    size = 10
    fill_value = 42.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 11: 1D float32 with size exactly matching unique count
    x = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    size = 5
    fill_value = 99.9
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_all_3"] = unique_all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_all_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_all_3'.")


check_valid('jax.numpy.unique_all', generated_inputs['jax.numpy.unique_all_3'], lib="jax", suffix=3)
