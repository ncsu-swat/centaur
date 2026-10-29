
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D, C order, boolean copy True
    a = np.arange(6, dtype=np.int32)
    shape = (2, 3)
    order = "C"
    copy_val = True
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 2: 2D to 1D, F order, boolean copy False
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    shape = (6,)
    order = "F"
    copy_val = False
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 3: Using -1 in shape inference
    a = np.zeros((3, 4, 2), dtype=np.float64)
    shape = (-1, 8)
    order = "C"
    copy_val = False
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 4: 0-D to 1-D with size 1
    a = np.array(42, dtype=np.int64)
    shape = (1,)
    order = "C"
    copy_val = True
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 5: High dimensional reshape
    a = np.ones((2, 2, 2, 2), dtype=np.complex64)
    shape = (4, 4)
    order = "F"
    copy_val = True
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 6: Reshape to 0-D (scalar shape)
    a = np.array([3.14], dtype=np.float32)
    shape = ()
    order = "C"
    copy_val = False
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 7: Empty array reshape
    a = np.empty((0, 5), dtype=np.int8)
    shape = (0, 10)
    order = "C"
    copy_val = False
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 8: Reshape with multiple 1-dimensions
    a = np.arange(12, dtype=np.uint8)
    shape = (1, 3, 1, 4, 1)
    order = "F"
    copy_val = True
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 9: Boolean tensor
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    shape = (4, 1)
    order = "C"
    copy_val = False
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    # Input 10: 3D to 3D with -1 inference and F order
    a = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    shape = (4, -1, 2)
    order = "F"
    copy_val = True
    list_of_inputs.append(
        {"a": a, "shape": shape, "order": order, "copy": copy_val}
    )

    return list_of_inputs


generated_inputs["jax.numpy.reshape_2"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reshape_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reshape_2'.")


check_valid('jax.numpy.reshape', generated_inputs['jax.numpy.reshape_2'], lib="jax", suffix=2)
