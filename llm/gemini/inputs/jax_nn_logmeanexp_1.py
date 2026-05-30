
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logmeanexp_inputs():
    list_of_inputs = []

    # Input 1, valid — 1D float32
    x = np.random.randn(10).astype(np.float32)
    axis = 0
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid — 2D float32, keepdims True
    x = np.random.randn(4, 4).astype(np.float32)
    axis = 1
    where = np.random.choice([True, False], size=x.shape)
    keepdims = True
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid — 3D float64
    x = np.random.randn(3, 4, 5).astype(np.float64)
    axis = 2
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid — 2D float32, axis 0
    x = np.random.randn(2, 3).astype(np.float32)
    axis = 0
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid — 2D with negative index axis
    x = np.random.randn(5, 5).astype(np.float32) * 10.0
    axis = -1
    where = np.random.choice([True, False], size=x.shape)
    keepdims = True
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid — 1D with manual mask
    x = np.random.randn(8).astype(np.float32)
    axis = 0
    where = np.array([True, True, False, True, False, True, True, True], dtype=bool)
    keepdims = False
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid — 4D float32
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = 2
    where = np.ones_like(x, dtype=bool)
    keepdims = True
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid — 3D float32, reduction along axis 0
    x = np.random.randn(4, 2, 3).astype(np.float32)
    axis = 0
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid — 2D float64, keepdims True
    x = np.random.randn(10, 2).astype(np.float64)
    axis = 1
    where = np.ones_like(x, dtype=bool)
    keepdims = True
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid — 3D float32, sparse mask
    x = np.random.randn(3, 3, 3).astype(np.float32)
    axis = 1
    where = np.random.choice([True, False], size=x.shape)
    keepdims = False
    input_dict = {"x": x, "axis": axis, "where": where, "keepdims": keepdims}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
