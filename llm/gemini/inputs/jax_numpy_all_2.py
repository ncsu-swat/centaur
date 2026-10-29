
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def all_inputs():
    list_of_inputs = []

    # Input 1: 2D boolean array, reducing axis 0
    a = np.array(
        [[True, True, False], [True, True, True]], dtype=bool
    )
    axis = (0,)
    keepdims = False
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 2: 2D float32 array with negative and positive values, keepdims True
    a = np.array([[-1.0, 2.5, 0.0], [3.2, -4.1, 5.0]], dtype=np.float32)
    axis = (1,)
    keepdims = True
    where = np.array([[True, True, False], [True, True, True]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 3: 1D int32 array
    a = np.array([1, 2, 0, -3, 4], dtype=np.int32)
    axis = (0,)
    keepdims = False
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 4: 3D int64 array, reducing multiple axes (0, 2)
    a = np.random.randint(-5, 5, size=(2, 3, 4), dtype=np.int64)
    axis = (0, 2)
    keepdims = True
    where = np.random.choice([True, False], size=(2, 3, 4))
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 5: 4D float64 array, broadcastable where mask
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    axis = (1, 3)
    keepdims = False
    where = np.ones((1, 3, 1, 5), dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 6: 2D complex64 array, negative axis index
    a = np.array(
        [[1.0 + 2.0j, 0.0 + 0.0j], [-1.0 - 1.0j, 2.0 + 0.0j]],
        dtype=np.complex64,
    )
    axis = (-1,)
    keepdims = True
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 7: 3D boolean array, reducing all axes
    a = np.full((3, 2, 2), True, dtype=bool)
    axis = (0, 1, 2)
    keepdims = False
    where = np.full((3, 2, 2), True, dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 8: 2D uint8 array, broadcastable where mask with keepdims
    a = np.array([[0, 255, 128], [1, 2, 3]], dtype=np.uint8)
    axis = (-2,)
    keepdims = True
    where = np.array([[True, True, False]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 9: 1D float32 array, negative axis
    a = np.array([-10.5, 0.0, 15.2], dtype=np.float32)
    axis = (-1,)
    keepdims = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 10: 3D float32 array, reducing axis 1
    a = np.random.uniform(-1.0, 1.0, size=(4, 5, 2)).astype(np.float32)
    axis = (1,)
    keepdims = True
    where = np.ones((4, 5, 2), dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    # Input 11: 2D int16 array with all non-zero values
    a = np.array([[-10, 20], [-30, -40]], dtype=np.int16)
    axis = (0,)
    keepdims = False
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append(
        {"a": a, "axis": axis, "keepdims": keepdims, "where": where}
    )

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.all_2"] = all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.all_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.all_2'.")


check_valid('jax.numpy.all', generated_inputs['jax.numpy.all_2'], lib="jax", suffix=2)
