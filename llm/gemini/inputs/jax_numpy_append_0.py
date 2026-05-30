
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def append_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, axis 0, float32
    arr = np.random.randn(5).astype(np.float32)
    values = np.random.randn(3).astype(np.float32)
    input_dict = {"arr": arr, "values": values, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, axis 0, int32
    arr = np.random.randint(-10, 10, size=(2, 3)).astype(np.int32)
    values = np.random.randint(-10, 10, size=(4, 3)).astype(np.int32)
    input_dict = {"arr": arr, "values": values, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, axis 1, float64
    arr = np.random.randn(3, 2).astype(np.float64)
    values = np.random.randn(3, 5).astype(np.float64)
    input_dict = {"arr": arr, "values": values, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, axis 2, float32
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    values = np.random.randn(2, 3, 1).astype(np.float32)
    input_dict = {"arr": arr, "values": values, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, negative axis -2, int64
    arr = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int64)
    values = np.random.randint(-100, 100, size=(2, 1, 4)).astype(np.int64)
    input_dict = {"arr": arr, "values": values, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D arrays, axis 0, float16
    arr = np.random.randn(1, 2, 2, 2).astype(np.float16)
    values = np.random.randn(3, 2, 2, 2).astype(np.float16)
    input_dict = {"arr": arr, "values": values, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D arrays, negative axis -1, float32
    arr = np.random.randn(10).astype(np.float32)
    values = np.random.randn(5).astype(np.float32)
    input_dict = {"arr": arr, "values": values, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D arrays, axis 1, boolean
    arr = np.random.choice([True, False], size=(4, 4))
    values = np.random.choice([True, False], size=(4, 2))
    input_dict = {"arr": arr, "values": values, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D arrays, axis 0, uint8
    arr = np.random.randint(0, 255, size=(5, 2, 3)).astype(np.uint8)
    values = np.random.randint(0, 255, size=(2, 2, 3)).astype(np.uint8)
    input_dict = {"arr": arr, "values": values, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays, axis 3, int16
    arr = np.random.randint(-50, 50, size=(2, 2, 2, 2)).astype(np.int16)
    values = np.random.randint(-50, 50, size=(2, 2, 2, 5)).astype(np.int16)
    input_dict = {"arr": arr, "values": values, "axis": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.append"] = append_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.append' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.append'.")


check_valid('jax.numpy.append', generated_inputs['jax.numpy.append'], lib="jax", suffix=0)
