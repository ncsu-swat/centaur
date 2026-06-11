
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_reduce_min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis (0,)
    operand = np.array([-1.5, 2.3, -0.1, 4.5, -10.2], dtype=np.float32)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis (1,)
    operand = np.random.randn(3, 5).astype(np.float32)
    axes = (1,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 array with negative/positive values, axis (0,)
    operand = np.random.randint(-50, 50, size=(4, 4)).astype(np.int32)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, axis (0, 2)
    operand = np.random.uniform(-10, 10, size=(2, 3, 4)).astype(np.float64)
    axes = (0, 2)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, axis (1,)
    operand = np.random.randint(-100, 100, size=(3, 2, 5)).astype(np.int64)
    axes = (1,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, axis (1, 2, 3)
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axes = (1, 2, 3)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D boolean array, axis (0,)
    operand = np.random.choice([True, False], size=(3, 3))
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float16 array, axis (0,)
    operand = np.array([1.0, -2.0, 3.5, -4.2], dtype=np.float16)
    axes = (0,)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D int32 array, axis (2, 4)
    operand = np.random.randint(0, 10, size=(2, 2, 3, 2, 4)).astype(np.int32)
    axes = (2, 4)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array, empty tuple (0 axes)
    operand = np.random.randn(2, 2, 2).astype(np.float32)
    axes = ()
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float32 array, reducing all axes (0, 1, 2)
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axes = (0, 1, 2)
    input_dict = {"operand": operand, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.reduce_min_1"] = generate_reduce_min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_min_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_min_1'.")


check_valid('jax.lax.reduce_min', generated_inputs['jax.lax.reduce_min_1'], lib="jax", suffix=1)
