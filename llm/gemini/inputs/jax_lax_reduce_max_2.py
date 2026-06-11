
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_reduce_max_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce over axis 0
    operand = np.array([1.0, -2.0, 3.5, -4.1, 5.0], dtype=np.float32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 2: 2D float32 array, reduce over axis 0
    operand = np.random.randn(3, 4).astype(np.float32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 3: 2D float32 array, reduce over axis 1
    operand = np.random.randn(4, 5).astype(np.float32)
    axes = (1,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 4: 2D float64 array, reduce over axes (0, 1)
    operand = np.random.randn(5, 5).astype(np.float64)
    axes = (0, 1)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 5: 3D int32 array (with negative numbers), reduce over axes (1, 2)
    operand = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    axes = (1, 2)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 6: 3D float32 array, reduce over axis 0
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 7: 4D int64 array, reduce over axes (2, 3)
    operand = np.random.randint(-1000, 1000, size=(2, 2, 3, 3)).astype(np.int64)
    axes = (2, 3)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 8: 3D float32 array, reduce over empty axes tuple
    operand = np.random.randn(3, 3, 3).astype(np.float32)
    axes = ()
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 9: 5D float32 array, reduce over axes (1, 3, 4)
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axes = (1, 3, 4)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 10: 2D int32 array, reduce over axis 1
    operand = np.arange(12).reshape(3, 4).astype(np.int32)
    axes = (1,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 11: 1D int64 array with negatives, reduce over axis 0
    operand = np.array([-10, -20, -3, -40], dtype=np.int64)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    return list_of_inputs

generated_inputs["jax.lax.reduce_max_2"] = generate_reduce_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_max_2'.")


check_valid('jax.lax.reduce_max', generated_inputs['jax.lax.reduce_max_2'], lib="jax", suffix=2)
