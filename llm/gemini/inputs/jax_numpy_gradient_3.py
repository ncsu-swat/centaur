
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D array, scalar spacing, gradient along axis 0
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=np.float32)
    varargs = np.array(1.0, dtype=np.float32)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, non-uniform coordinate spacing
    f = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0], dtype=np.float64)
    varargs = np.array([0.0, 1.0, 1.5, 3.0, 4.0, 6.0, 8.5, 10.0], dtype=np.float64)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, gradient along axis 0 with scalar spacing
    f = np.random.randn(5, 6).astype(np.float32)
    varargs = np.array(0.5, dtype=np.float32)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, gradient along axis 1 with 1D coordinate array
    f = np.random.randn(4, 7).astype(np.float64)
    varargs = np.linspace(0.0, 10.0, 7, dtype=np.float64)
    axis = (1,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, negative axis index
    f = np.arange(24, dtype=np.float32).reshape(4, 6)
    varargs = np.array(2.0, dtype=np.float32)
    axis = (-1,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, gradient along multiple axes with scalar spacing
    f = np.random.randn(8, 8).astype(np.float32)
    varargs = np.array(0.25, dtype=np.float32)
    axis = (0, 1)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with negative values, gradient along axis 1
    f = np.linspace(-10.0, 10.0, 60, dtype=np.float32).reshape(3, 4, 5)
    varargs = np.array(1.5, dtype=np.float32)
    axis = (1,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, coordinate spacing along axis 2
    f = np.random.uniform(-1.0, 1.0, size=(4, 3, 6)).astype(np.float64)
    varargs = np.array([0.0, 0.1, 0.4, 0.9, 1.6, 2.5], dtype=np.float64)
    axis = (2,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, gradient along multiple axes (0, 2)
    f = np.random.randn(5, 5, 5).astype(np.float32)
    varargs = np.array(0.1, dtype=np.float32)
    axis = (0, 2)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, gradient along single axis 3
    f = np.random.randn(2, 3, 4, 5).astype(np.float32)
    varargs = np.array(1.0, dtype=np.float32)
    axis = (3,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array, gradient along all axes with negative spacing
    f = np.ones((3, 3, 3, 3), dtype=np.float64)
    varargs = np.array(-0.5, dtype=np.float64)
    axis = (0, 1, 2, 3)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_3"] = gradient_inputs()

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
