
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_ptp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, keepdims False
    a = np.array([1.0, 5.0, 3.0, 10.0, -2.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative values, axis 1, keepdims True
    a = np.array([[1, 3, 5], [-4, 6, 8], [7, -9, 3]], dtype=np.int32)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, axis 2, keepdims False
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"a": a, "axis": 2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, axis 0, keepdims True
    a = np.random.uniform(-10.0, 10.0, size=(5, 5)).astype(np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D int64 array, negative axis, keepdims False
    a = np.array([100, -200, 300, 0, -50], dtype=np.int64)
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, axis 1, keepdims True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int16 array, axis 0, keepdims False
    a = np.random.randint(-100, 100, size=(3, 3, 3)).astype(np.int16)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, axis -2, keepdims True
    a = np.random.randn(10, 2).astype(np.float32)
    input_dict = {"a": a, "axis": -2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float64 array, axis 4, keepdims False
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {"a": a, "axis": 4, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int32 array, axis 1, keepdims False
    a = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ptp_1"] = jax_numpy_ptp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ptp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ptp_1'.")


check_valid('jax.numpy.ptp', generated_inputs['jax.numpy.ptp_1'], lib="jax", suffix=1)
