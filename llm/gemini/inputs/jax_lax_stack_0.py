
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def stack_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array representing a sequence of 3 1D arrays of shape (4,), axis=0
    operands = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"operands": operands, "axis": 0})

    # Input 2: 2D int32 array representing a sequence of 2 1D arrays of shape (5,), axis=1
    operands = np.random.randint(-10, 10, size=(2, 5)).astype(np.int32)
    list_of_inputs.append({"operands": operands, "axis": 1})

    # Input 3: 3D float64 array representing a sequence of 4 2D arrays of shape (2, 3), axis=0
    operands = np.random.randn(4, 2, 3).astype(np.float64)
    list_of_inputs.append({"operands": operands, "axis": 0})

    # Input 4: 3D bool array representing a sequence of 2 2D arrays of shape (2, 2), axis=-1
    operands = np.random.choice([True, False], size=(2, 2, 2)).astype(bool)
    list_of_inputs.append({"operands": operands, "axis": -1})

    # Input 5: 4D int16 array representing a sequence of 3 3D arrays of shape (2, 2, 2), axis=2
    operands = np.random.randint(-100, 100, size=(3, 2, 2, 2)).astype(np.int16)
    list_of_inputs.append({"operands": operands, "axis": 2})

    # Input 6: 5D float32 array representing a sequence of 2 4D arrays of shape (1, 2, 2, 1), axis=-2
    operands = np.random.randn(2, 1, 2, 2, 1).astype(np.float32)
    list_of_inputs.append({"operands": operands, "axis": -2})

    # Input 7: 1D float32 array representing a sequence of 3 0D arrays (scalars), axis=0
    operands = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"operands": operands, "axis": 0})

    # Input 8: 3D uint8 array representing a sequence of 3 2D arrays of shape (3, 3), axis=1
    operands = np.random.randint(0, 255, size=(3, 3, 3)).astype(np.uint8)
    list_of_inputs.append({"operands": operands, "axis": 1})

    # Input 9: 4D float32 array representing a sequence of 2 3D arrays of shape (2, 2, 2), axis=-3
    operands = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"operands": operands, "axis": -3})

    # Input 10: 2D float32 array representing a sequence of 2 1D arrays of shape (1,), axis=-1
    operands = np.random.randn(2, 1).astype(np.float32)
    list_of_inputs.append({"operands": operands, "axis": -1})

    return list_of_inputs

generated_inputs["jax.lax.stack"] = stack_inputs()

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
