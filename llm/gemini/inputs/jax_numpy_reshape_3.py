
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D, C-order
    a = np.arange(6, dtype=np.int32)
    shape = [2, 3]
    order = "C"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 2: 2D to 1D, F-order
    a = np.ones((3, 4), dtype=np.float32)
    shape = [12]
    order = "F"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 3: Using -1 in shape dimension
    a = np.zeros((2, 3, 4), dtype=np.float64)
    shape = [-1, 6]
    order = "C"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 4: 3D to 3D with different dimensions, F-order
    a = np.random.randn(2, 4, 3).astype(np.float32)
    shape = [4, 3, 2]
    order = "F"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 5: Scalar to 1D array
    a = np.array(42, dtype=np.int64)
    shape = [1]
    order = "C"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 6: Higher dimension (4D to 2D)
    a = np.arange(24, dtype=np.uint8).reshape((2, 3, 2, 2))
    shape = [6, 4]
    order = "C"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 7: Empty array reshape
    a = np.empty((0, 5), dtype=np.float32)
    shape = [0, 10]
    order = "C"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 8: Complex data type
    a = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
    shape = [2, 2]
    order = "F"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 9: Boolean tensor
    a = np.array([[True, False], [False, True]], dtype=bool)
    shape = [-1]
    order = "C"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 10: 5D tensor reshape to 3D
    a = np.ones((2, 1, 2, 1, 3), dtype=np.int16)
    shape = [2, 2, 3]
    order = "F"
    copy_flag = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    # Input 11: Reshaping to scalar equivalent empty list shape
    a = np.array([3.14], dtype=np.float32)
    shape = []
    order = "C"
    copy_flag = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shape": shape, "order": order, "copy": copy_flag})
    )

    return list_of_inputs


generated_inputs["jax.numpy.reshape_3"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reshape_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reshape_3'.")


check_valid('jax.numpy.reshape', generated_inputs['jax.numpy.reshape_3'], lib="jax", suffix=3)
