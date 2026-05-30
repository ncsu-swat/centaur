
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_all_inputs():
    list_of_inputs = []

    # Input 1: 1D bool array, axis [0], keepdims=False
    a = np.array([True, True, False, True], dtype=bool)
    axis = [0]
    keepdims = False
    where = np.array([True, True, False, True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D bool array, axis [1], keepdims=True
    a = np.array([[True, False], [True, True]], dtype=bool)
    axis = [1]
    keepdims = True
    where = np.array([[True, True], [True, False]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D bool array, axis [0, 2], keepdims=False
    a = np.random.choice([True, False], size=(2, 3, 4))
    axis = [0, 2]
    keepdims = False
    where = np.random.choice([True, False], size=(2, 3, 4))
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float array (numeric values act as bool), axis [0], keepdims=True
    a = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    axis = [0]
    keepdims = True
    where = np.array([[True, True], [True, True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int array, negative axis [-1], keepdims=False
    a = np.ones((2, 2, 2), dtype=np.int32)
    axis = [-1]
    keepdims = False
    where = np.ones((2, 2, 2), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D bool array, axis [0], keepdims=True, broadcastable 1-element 'where'
    a = np.array([True, False, True], dtype=bool)
    axis = [0]
    keepdims = True
    where = np.array([True], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D bool array, axis [1, 3], keepdims=False
    a = np.random.choice([True, False], size=(2, 2, 2, 2))
    axis = [1, 3]
    keepdims = False
    where = np.random.choice([True, False], size=(2, 2, 2, 2))
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D bool array, negative axis [-2], keepdims=True
    a = np.array([[False, False], [False, False]], dtype=bool)
    axis = [-2]
    keepdims = True
    where = np.array([[False, True], [True, False]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D bool array, axis [0, 1, 2], keepdims=True
    a = np.random.choice([True, False], size=(3, 3, 3))
    axis = [0, 1, 2]
    keepdims = True
    where = np.ones((3, 3, 3), dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D bool array, axis [0, 1], keepdims=False, 2D broadcastable 'where'
    a = np.array([[True, True], [True, True]], dtype=bool)
    axis = [0, 1]
    keepdims = False
    where = np.array([[True]], dtype=bool)
    input_dict = {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.all_3"] = jax_numpy_all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.all_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.all_3'.")


check_valid('jax.numpy.all', generated_inputs['jax.numpy.all_3'], lib="jax", suffix=3)
