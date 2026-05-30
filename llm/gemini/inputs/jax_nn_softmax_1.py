
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis -1, where is all True
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = -1
    where = np.array([True, True, True, True], dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis 0, where is mask
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 0
    where = np.array([[True, False], [True, True]], dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 1, where is all True
    x = np.random.randn(3, 5).astype(np.float64)
    axis = 1
    where = np.ones((3, 5), dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, axis 2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    where = np.ones((2, 3, 4), dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, axis -1
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = -1
    where = np.ones((2, 2, 2, 2), dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array with negative values, axis 1
    x = np.array([[-10.0, -20.0, 0.0], [5.0, -5.0, 10.0]], dtype=np.float32)
    axis = 1
    where = np.array([[True, True, False], [True, True, True]], dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array, axis 0, partial where mask
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    axis = 0
    where = np.array([False, True, True], dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, axis 1, where has some False
    x = np.random.uniform(-1, 1, (2, 4, 3)).astype(np.float32)
    axis = 1
    where = (np.random.rand(2, 4, 3) > 0.2)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array with shape 5x5, axis 0
    x = np.random.randn(5, 5).astype(np.float32)
    axis = 0
    where = np.ones((5, 5), dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array, axis -1, where has single active element
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    axis = -1
    where = np.array([False, True, False], dtype=bool)
    input_dict = {"x": x, "axis": axis, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.softmax_1"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.softmax_1'.")


check_valid('jax.nn.softmax', generated_inputs['jax.nn.softmax_1'], lib="jax", suffix=1)
