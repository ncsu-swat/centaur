
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argmin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, keepdims False
    a = np.array([3.5, 1.2, -0.4, 2.8, -1.5, 0.0], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative axis, keepdims True
    a = np.array([-10, 20, -30, 40, -50], dtype=np.int32)
    input_dict = {"a": a, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 0, keepdims True
    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array, axis 1, keepdims False
    a = np.random.randint(-100, 100, size=(5, 4), dtype=np.int64)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array with negative axis, keepdims False
    a = np.random.randn(3, 7).astype(np.float32)
    input_dict = {"a": a, "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 2, keepdims True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int16 array, axis -1, keepdims False
    a = np.random.randint(-50, 50, size=(4, 2, 6), dtype=np.int16)
    input_dict = {"a": a, "axis": -1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float64 array, axis 1, keepdims True
    a = np.random.randn(2, 3, 2, 4).astype(np.float64)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D uint8 array, axis 3, keepdims False
    a = np.random.randint(0, 256, size=(2, 2, 3, 5), dtype=np.uint8)
    input_dict = {"a": a, "axis": 3, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D boolean array, axis 0, keepdims True
    a = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float32 array with duplicate minimum values, axis 1, keepdims False
    a = np.zeros((3, 4, 2), dtype=np.float32)
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
