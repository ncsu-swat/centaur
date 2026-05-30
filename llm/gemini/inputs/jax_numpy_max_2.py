
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def max_inputs():
    list_of_inputs = []

    # Input 1: 1D array
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([True, True, False, True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=1, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    keepdims = True
    initial = np.array(-5.0, dtype=np.float32)
    where = np.random.choice([True, False], size=(3, 4)).astype(bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, multi-axis reduction
    a = np.random.randn(3, 4).astype(np.float32)
    axis = (0, 1)
    keepdims = False
    initial = np.array(-100.0, dtype=np.float32)
    where = np.ones((3, 4), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, integer dtype
    a = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int32)
    axis = (2,)
    keepdims = False
    initial = np.array(-200, dtype=np.int32)
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, broadcasted where
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    keepdims = True
    initial = np.array(-1.0, dtype=np.float32)
    where = np.ones((1, 3, 1), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, float64
    a = np.random.randn(5, 5).astype(np.float64)
    axis = (0,)
    keepdims = False
    initial = np.array(-10.0, dtype=np.float64)
    where = np.random.choice([True, False], size=(5, 5)).astype(bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: negative integer array
    a = np.array([[-5, -10], [-3, -15]], dtype=np.int32)
    axis = (1,)
    keepdims = True
    initial = np.array(-20, dtype=np.int32)
    where = np.array([[True, False], [True, True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (1, 3)
    keepdims = False
    initial = np.array(-5.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D integer array with partially True where mask
    a = np.array([10, -20, 30, -40], dtype=np.int32)
    axis = (0,)
    keepdims = True
    initial = np.array(-50, dtype=np.int32)
    where = np.array([True, True, True, False], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with all False where (initial will be returned)
    a = np.random.randn(2, 2).astype(np.float32)
    axis = (0, 1)
    keepdims = False
    initial = np.array(99.0, dtype=np.float32)
    where = np.zeros((2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "initial": initial, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.max_2"] = max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.max_2'.")


check_valid('jax.numpy.max', generated_inputs['jax.numpy.max_2'], lib="jax", suffix=2)
