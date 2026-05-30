
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gradient_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, 0D spacing, axis (0,), edge_order None
    f = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float32)
    varargs = np.array(1.0, dtype=np.float32)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, 1D coordinates, axis (0,), edge_order None
    f = np.array([1.0, 2.0, 4.0, 7.0], dtype=np.float64)
    varargs = np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, 0D spacing, axis (0,), edge_order None
    f = np.random.randn(4, 5).astype(np.float32)
    varargs = np.array(0.5, dtype=np.float32)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, 1D coordinates matching axis 1, axis (1,), edge_order None
    f = np.random.randn(3, 4).astype(np.float64)
    varargs = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64)
    axis = (1,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, 0D spacing, axis (1,), edge_order None
    f = np.random.randn(2, 3, 4).astype(np.float32)
    varargs = np.array(2.5, dtype=np.float32)
    axis = (1,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int32 array, 0D spacing, axis (0,), edge_order None
    f = np.array([1, 3, 6, 10], dtype=np.int32)
    varargs = np.array(1, dtype=np.int32)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array, 0D spacing, axis (1,), edge_order None
    f = np.random.randn(5, 5).astype(np.float32)
    varargs = np.array(-1.5, dtype=np.float32)
    axis = (1,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array with negative values, 1D coordinates, axis (0,), edge_order None
    f = np.array([-1.0, -2.0, -4.0, -7.0], dtype=np.float32)
    varargs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array, 0D spacing, axis (2,), edge_order None
    f = np.random.randn(2, 2, 3).astype(np.float64)
    varargs = np.array(0.1, dtype=np.float64)
    axis = (2,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, 1D coordinates matching axis 0, axis (0,), edge_order None
    f = np.random.randn(3, 2).astype(np.float32)
    varargs = np.array([1.0, 1.5, 3.0], dtype=np.float32)
    axis = (0,)
    edge_order = None
    input_dict = {"f": f, "varargs": varargs, "axis": axis, "edge_order": edge_order}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.gradient_2"] = gradient_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gradient_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gradient_2'.")


check_valid('jax.numpy.gradient', generated_inputs['jax.numpy.gradient_2'], lib="jax", suffix=2)
