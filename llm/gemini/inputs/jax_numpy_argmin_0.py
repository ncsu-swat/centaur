
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argmin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, keepdims False
    a = np.array([3.0, 1.0, 2.0, 0.5, 4.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis 1, keepdims True
    a = np.array([[5.0, 2.0, 9.0], [1.0, 7.0, 3.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array, axis 0, keepdims False
    a = np.array([[10, 20], [5, 30], [15, 2]], dtype=np.int32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, axis -1, keepdims True
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, axis 2, keepdims False
    a = np.random.randint(-100, 100, size=(3, 2, 5)).astype(np.int64)
    input_dict = {"a": a, "axis": 2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, axis 1, keepdims True
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array with negative numbers, axis -1, keepdims False
    a = np.array([-10, -50, 0, 20, -50, 30], dtype=np.int32)
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, negative axis, keepdims True
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {"a": a, "axis": -2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D int16 array, axis 3, keepdims False
    a = np.random.randint(-10, 10, size=(2, 2, 2, 3, 2)).astype(np.int16)
    input_dict = {"a": a, "axis": 3, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 array with duplicate minimums, axis 1, keepdims False
    a = np.array([[1.0, 1.0, 2.0], [3.0, 0.5, 0.5]], dtype=np.float64)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argmin"] = argmin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argmin'.")


check_valid('jax.numpy.argmin', generated_inputs['jax.numpy.argmin'], lib="jax", suffix=0)
