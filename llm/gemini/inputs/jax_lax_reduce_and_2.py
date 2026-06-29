
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_reduce_and_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean, reduce over axis 0
    operand = np.array([True, True, False, True], dtype=bool)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 2: 2D int32, reduce over axis 0
    operand = np.array([[5, 3], [7, 3]], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 3: 2D int32 with negative numbers, reduce over axis 1
    operand = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    axes = (1,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 4: 3D boolean, reduce over axes 0 and 2
    operand = np.random.choice([True, False], size=(2, 3, 4)).astype(bool)
    axes = (0, 2)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 5: 2D int32, reduce over axis 1
    operand = np.array([[12, 15, 14], [15, 15, 15]], dtype=np.int32)
    axes = (1,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 6: 4D int32, reduce over multiple non-contiguous axes
    operand = np.random.randint(-100, 100, size=(2, 2, 2, 2)).astype(np.int32)
    axes = (1, 3)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 7: 1D int32, large values
    operand = np.array([1023, 511, 255], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 8: 2D boolean, empty axes tuple
    operand = np.array([[True, False], [False, True]], dtype=bool)
    axes = ()
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 9: 3D int32, reduce all axes
    operand = np.ones((2, 2, 2), dtype=np.int32) * 255
    axes = (0, 1, 2)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 10: 3D int32, reduce over axis 2
    operand = np.random.randint(-1000, 1000, size=(3, 4, 5)).astype(np.int32)
    axes = (2,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    return list_of_inputs

generated_inputs["jax.lax.reduce_and_2"] = generate_reduce_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_and_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_and_2'.")


check_valid('jax.lax.reduce_and', generated_inputs['jax.lax.reduce_and_2'], lib="jax", suffix=2)
