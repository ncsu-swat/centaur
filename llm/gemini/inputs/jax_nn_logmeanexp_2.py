
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_logmeanexp_inputs():
    list_of_inputs = []

    # Case 1: 1D array, float32, keepdims=False
    x = np.random.randn(5).astype(np.float32)
    axis = (0,)
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 2: 2D array, float32, keepdims=True
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    where = np.ones_like(x, dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 3: 3D array, float64, keepdims=False
    x = np.random.randn(2, 3, 4).astype(np.float64)
    axis = (0, 2)
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 4: 2D array, float32, with where mask, keepdims=True
    x = np.random.randn(4, 5).astype(np.float32)
    axis = (0,)
    where = np.random.choice([True, False], size=x.shape).astype(bool)
    where[0, :] = True  # Ensure at least some elements are True to avoid empty reduction
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 5: 4D array, float32, multiple axes, keepdims=False
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = (1, 3)
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 6: 1D array, float16, keepdims=True
    x = np.random.randn(10).astype(np.float16)
    axis = (0,)
    where = np.ones_like(x, dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 7: 3D array, float32, keepdims=False, with some masked values
    x = np.random.randn(2, 4, 3).astype(np.float32)
    axis = (1,)
    where = np.random.choice([True, False], size=x.shape).astype(bool)
    where[:, 0, :] = True  # Ensure no axis slice is completely masked out
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 8: 2D array, float64, reduce all axes, keepdims=True
    x = np.random.randn(6, 6).astype(np.float64)
    axis = (0, 1)
    where = np.ones_like(x, dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 9: 5D array, float32, keepdims=False
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = (2, 4)
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Case 10: 2D array, float32, large range, keepdims=False
    x = (np.random.randn(5, 5) * 10).astype(np.float32)
    axis = (1,)
    where = np.ones_like(x, dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    return list_of_inputs

generated_inputs["jax.nn.logmeanexp_2"] = jax_nn_logmeanexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logmeanexp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logmeanexp_2'.")


check_valid('jax.nn.logmeanexp', generated_inputs['jax.nn.logmeanexp_2'], lib="jax", suffix=2)
