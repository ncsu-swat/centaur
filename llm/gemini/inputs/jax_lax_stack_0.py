
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_stack_inputs():
    list_of_inputs = []

    # Input 1: 2 arrays of shape (3,), float32, axis=0
    operands = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"operands": operands, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3 arrays of shape (2,), int32, axis=1
    operands = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    input_dict = {"operands": operands, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2 arrays of shape (3,), float64, negative axis=-1
    operands = np.array(
        [[-1.5, 0.5, 2.5], [3.5, -4.5, 5.5]], dtype=np.float64
    )
    input_dict = {"operands": operands, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3 scalars of shape (), int64, axis=0
    operands = np.array([42, -7, 0], dtype=np.int64)
    input_dict = {"operands": operands, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2 arrays of shape (2, 3), float32, axis=0
    operands = np.ones((2, 2, 3), dtype=np.float32)
    input_dict = {"operands": operands, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2 arrays of shape (2, 3), int32, axis=1
    operands = np.arange(12, dtype=np.int32).reshape((2, 2, 3))
    input_dict = {"operands": operands, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3 arrays of shape (3, 4), float32, negative axis=-2
    operands = np.random.randn(3, 3, 4).astype(np.float32)
    input_dict = {"operands": operands, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2 arrays of shape (2, 3, 4), float32, axis=2
    operands = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {"operands": operands, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2 arrays of shape (2, 2, 2), int16, trailing axis=3
    operands = np.ones((2, 2, 2, 2), dtype=np.int16)
    input_dict = {"operands": operands, "axis": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2 boolean arrays of shape (2, 2), axis=0
    operands = np.array(
        [[[True, False], [False, True]], [[False, True], [True, False]]],
        dtype=np.bool_,
    )
    input_dict = {"operands": operands, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2 complex64 arrays of shape (2,), axis=1
    operands = np.array(
        [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
        dtype=np.complex64,
    )
    input_dict = {"operands": operands, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2 uint8 arrays of shape (1, 2, 3, 4), axis=-1
    operands = np.zeros((2, 1, 2, 3, 4), dtype=np.uint8)
    input_dict = {"operands": operands, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.stack"] = lax_stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.stack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.stack'.")


check_valid('jax.lax.stack', generated_inputs['jax.lax.stack'], lib="jax", suffix=0)
