
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logmeanexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, keepdims=False
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = 0
    where = np.array([True, True, True], dtype=bool)
    keepdims = False
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 2: 2D array, axis=0, keepdims=True
    x = np.random.randn(4, 5).astype(np.float32)
    axis = 0
    where = np.ones((4, 5), dtype=bool)
    keepdims = True
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 3: 2D array, axis=1, keepdims=False, with some False in where
    x = np.random.randn(3, 4).astype(np.float64)
    axis = 1
    where = np.array([
        [True, True, False, True],
        [False, True, True, True],
        [True, False, True, False]
    ], dtype=bool)
    keepdims = False
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 4: 2D array, negative axis=-1, keepdims=True
    x = np.array([[-10.0, -20.0], [-30.0, -40.0]], dtype=np.float32)
    axis = -1
    where = np.array([[True, True], [True, False]], dtype=bool)
    keepdims = True
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 5: 3D array, axis=1, broadcasted where
    x = np.random.uniform(0.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    axis = 1
    where = np.ones((2, 3, 1), dtype=bool)
    keepdims = False
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 6: 3D array, axis=2, keepdims=True
    x = np.random.randn(3, 2, 5).astype(np.float32)
    axis = 2
    where = np.random.choice([True, False], size=(3, 2, 5), p=[0.8, 0.2])
    where[:, :, 0] = True
    keepdims = True
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 7: 1D array with negative numbers, axis=-1, keepdims=True
    x = np.array([-1.5, -2.5, -0.5, -100.0], dtype=np.float64)
    axis = -1
    where = np.array([True, True, True, False], dtype=bool)
    keepdims = True
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 8: 4D array, axis=2, keepdims=False
    x = np.random.randn(2, 3, 4, 2).astype(np.float32)
    axis = 2
    where = np.ones((2, 3, 4, 2), dtype=bool)
    keepdims = False
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 9: 2D array, axis=0, keepdims=False, float16
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    axis = 0
    where = np.array([[True, True], [True, True]], dtype=bool)
    keepdims = False
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 10: 3D array, axis=0, negative values and mixed where
    x = np.full((4, 2, 3), -5.0, dtype=np.float32)
    axis = 0
    where = np.ones((4, 2, 3), dtype=bool)
    where[0, :, :] = True
    keepdims = True
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    # Input 11: 4D array, axis=3, keepdims=False
    x = np.random.randn(1, 2, 3, 4).astype(np.float64)
    axis = 3
    where = np.ones((1, 2, 3, 4), dtype=bool)
    keepdims = False
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis, "where": where, "keepdims": keepdims}))

    return list_of_inputs

generated_inputs["jax.nn.logmeanexp_1"] = logmeanexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logmeanexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logmeanexp_1'.")


check_valid('jax.nn.logmeanexp', generated_inputs['jax.nn.logmeanexp_1'], lib="jax", suffix=1)
