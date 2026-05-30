
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, scalar spacing
    f = np.random.randn(10).astype(np.float32)
    varargs = np.array(2.0, dtype=np.float32)
    axis = (0,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 2: 1D array, float64, coordinate spacing
    f = np.random.randn(5).astype(np.float64)
    varargs = np.array([1.0, 3.0, 4.0, 7.0, 9.0], dtype=np.float64)
    axis = (0,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 3: 2D array, float32, scalar spacing, gradient along axis 0
    f = np.random.randn(4, 4).astype(np.float32)
    varargs = np.array(1.0, dtype=np.float32)
    axis = (0,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 4: 2D array, float32, coordinate spacing, gradient along axis 1
    f = np.random.randn(3, 5).astype(np.float32)
    varargs = np.array([0.1, 0.2, 0.5, 1.0, 2.0], dtype=np.float32)
    axis = (1,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 5: 3D array, float32, scalar spacing, gradient along axis 2
    f = np.random.randn(2, 3, 4).astype(np.float32)
    varargs = np.array(0.5, dtype=np.float32)
    axis = (2,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 6: 2D array with negative values, float32, scalar spacing
    f = -np.random.randn(5, 5).astype(np.float32)
    varargs = np.array(0.1, dtype=np.float32)
    axis = (0,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 7: Large 1D array, float64, scalar spacing
    f = np.random.randn(100).astype(np.float64)
    varargs = np.array(0.01, dtype=np.float64)
    axis = (0,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 8: 3D array, float32, coordinate spacing along axis 1
    f = np.random.randn(2, 4, 2).astype(np.float32)
    varargs = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    axis = (1,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 9: 4D array, float32, scalar spacing, gradient along axis 3
    f = np.random.randn(2, 2, 2, 2).astype(np.float32)
    varargs = np.array(1.5, dtype=np.float32)
    axis = (3,)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    # Input 10: 2D array, float32, scalar spacing for multiple axes
    f = np.random.randn(3, 3).astype(np.float32)
    varargs = np.array(2.0, dtype=np.float32)
    axis = (0, 1)
    edge_order = None
    list_of_inputs.append({
        "f": f,
        "varargs": varargs,
        "axis": axis,
        "edge_order": edge_order
    })

    return list_of_inputs

generated_inputs["jax.numpy.gradient_3"] = jax_numpy_gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_3'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_3'], lib="jax", suffix=3)
