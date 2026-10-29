
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D array, expand axis 0
    a = np.array([1, 2, 3], dtype=np.int32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 1D array, expand trailing axis
    a = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: 1D array, expand multiple axes
    a = np.array([10, 20, 30], dtype=np.int64)
    axis = [0, 1, 3]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 0D scalar array, expand to 1D
    a = np.array(42.0, dtype=np.float64)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: 0D scalar array, expand to multiple dimensions
    a = np.array(True, dtype=np.bool_)
    axis = [0, 1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 2D array, expand middle axis
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: 2D array, negative axis index
    a = np.ones((3, 4), dtype=np.float32)
    axis = [-1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: 2D array, multiple negative and positive axes
    a = np.zeros((2, 5), dtype=np.complex64)
    axis = [0, -1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 3D array, expand multiple distributed axes
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = [0, 2, 4]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: 4D array, expand leading and trailing dimensions
    a = np.ones((2, 2, 2, 2), dtype=np.uint8)
    axis = [0, 5]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: Empty array (size 0)
    a = np.empty((0, 3), dtype=np.float32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 12: 1D array, negative index expansion
    a = np.array([-1, 0, 1], dtype=np.int16)
    axis = [-2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.expand_dims_3"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expand_dims_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expand_dims_3'.")


check_valid('jax.numpy.expand_dims', generated_inputs['jax.numpy.expand_dims_3'], lib="jax", suffix=3)
