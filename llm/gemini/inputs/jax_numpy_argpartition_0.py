
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argpartition_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, standard partition
    a = np.array([6.0, 8.0, 4.0, 3.0, 1.0, 9.0, 7.0, 5.0, 2.0], dtype=np.float32)
    kth = 4
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 2: 1D array, int32, partition at 0 (minimum element)
    a = np.array([10, -5, 3, 0, -20, 8], dtype=np.int32)
    kth = 0
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 3: 2D array, float64, partition along axis 0
    a = np.array([[5.5, 2.1, 8.9], [1.2, 7.4, 0.3], [9.8, 3.6, 4.2]], dtype=np.float64)
    kth = 1
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 4: 2D array, int64, partition along axis 1
    a = np.arange(20, dtype=np.int64).reshape(4, 5)[:, ::-1]
    kth = 3
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 5: 3D array, float32, negative axis
    a = np.random.randn(2, 4, 6).astype(np.float32)
    kth = 2
    axis = -2
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 6: 1D array with duplicate values and negative numbers
    a = np.array([-3, 5, -3, 2, 2, -1, 0, 5], dtype=np.int32)
    kth = 5
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 7: 3D array, int16, axis = 2
    a = np.random.randint(-100, 100, size=(3, 3, 5), dtype=np.int16)
    kth = 4
    axis = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 8: 4D array, float32, axis = 1
    a = np.random.uniform(-10.0, 10.0, size=(2, 5, 3, 2)).astype(np.float32)
    kth = 2
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 9: 1D array, float16 dtype
    a = np.array([3.5, 1.2, 7.8, -2.4, 0.0, 4.1], dtype=np.float16)
    kth = 3
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    # Input 10: 2D array with inf and nan values
    a = np.array([[np.nan, 1.0, np.inf], [-np.inf, 2.0, 0.0]], dtype=np.float32)
    kth = 1
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "kth": kth, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.argpartition"] = argpartition_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argpartition' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argpartition'.")


check_valid('jax.numpy.argpartition', generated_inputs['jax.numpy.argpartition'], lib="jax", suffix=0)
