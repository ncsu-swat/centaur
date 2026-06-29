
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split into 2 equal parts, float32
    operand = np.random.randn(10).astype(np.float32)
    sizes = (5, 5)
    axis = 0
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, split along axis 0, unequal parts, float32
    operand = np.random.randn(6, 4).astype(np.float32)
    sizes = (1, 2, 3)
    axis = 0
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, split along axis 1, unequal parts, float64
    operand = np.random.randn(4, 10).astype(np.float64)
    sizes = (2, 3, 5)
    axis = 1
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, split along axis 0 with negative axis, float32
    operand = np.random.randn(3, 4, 5).astype(np.float32)
    sizes = (1, 2)
    axis = -3
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, split along axis 2, float32
    operand = np.random.randn(2, 2, 8).astype(np.float32)
    sizes = (4, 4)
    axis = 2
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, split along axis 2, int32
    operand = np.random.randint(-10, 10, size=(2, 3, 4, 5)).astype(np.int32)
    sizes = (1, 2, 1)
    axis = 2
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, split into 3 equal parts, int64
    operand = np.random.randint(-100, 100, size=(15,)).astype(np.int64)
    sizes = (5, 5, 5)
    axis = 0
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, split along axis 1, complex64
    operand = (np.random.randn(4, 6) + 1j * np.random.randn(4, 6)).astype(np.complex64)
    sizes = (2, 4)
    axis = 1
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, split along axis 4, float32
    operand = np.random.randn(1, 1, 1, 1, 9).astype(np.float32)
    sizes = (3, 3, 3)
    axis = 4
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, split along negative axis -2, float32
    operand = np.random.randn(2, 6, 2).astype(np.float32)
    sizes = (1, 2, 3)
    axis = -2
    input_dict = {"operand": operand, "sizes": sizes, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.split_2"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.split_2'.")


check_valid('jax.lax.split', generated_inputs['jax.lax.split_2'], lib="jax", suffix=2)
