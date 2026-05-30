
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_logmeanexp_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, True], dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 2: 2D array, reduction on axis 1, keeping dimensions
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    where = np.ones((3, 4), dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 3: 3D array with float64, reducing multiple axes
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float64)
    axis = (0, 2)
    where = np.ones((2, 3, 4), dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 4: Negative values in 2D array
    x = np.array([[-5.0, -10.0], [-15.0, -20.0]], dtype=np.float32)
    axis = (0,)
    where = np.array([[True, True], [True, True]], dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 5: 4D array with float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = (1, 3)
    where = np.ones((2, 2, 3, 3), dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 6: 1D array, large values, empty axis tuple (reducing no axes)
    x = np.array([100.0, 200.0, -50.0], dtype=np.float32)
    axis = ()
    where = np.ones((3,), dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 7: 2D array, fully reduced
    x = np.random.randn(5, 5).astype(np.float32)
    axis = (0, 1)
    where = np.ones((5, 5), dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 8: 3D float16
    x = np.random.randn(2, 2, 2).astype(np.float16)
    axis = (2,)
    where = np.ones((2, 2, 2), dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 9: 2D array with specific shape and masking
    x = np.arange(12, dtype=np.float32).reshape((3, 4))
    axis = (0,)
    where = np.ones((3, 4), dtype=bool)
    keepdims = False
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    # Input 10: 5D array
    x = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    axis = (2, 4)
    where = np.ones((2, 1, 3, 1, 2), dtype=bool)
    keepdims = True
    list_of_inputs.append({"x": x, "axis": axis, "where": where, "keepdims": keepdims})

    return list_of_inputs

generated_inputs["jax.nn.logmeanexp_3"] = jax_nn_logmeanexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logmeanexp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logmeanexp_3'.")


check_valid('jax.nn.logmeanexp', generated_inputs['jax.nn.logmeanexp_3'], lib="jax", suffix=3)
