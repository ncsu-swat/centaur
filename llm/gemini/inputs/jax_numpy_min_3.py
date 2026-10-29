
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis=[0], keepdims=False
    a = np.array([3.5, -1.2, 0.0, 7.8, -4.5], dtype=np.float32)
    axis = [0]
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 2: 2D int32 array with negative values, axis=[1], keepdims=True
    a = np.array(
        [[10, -20, 30], [-5, 15, -25], [0, 40, -10]], dtype=np.int32
    )
    axis = [1]
    keepdims = True
    initial = np.array(100, dtype=np.int32)
    where = np.array(
        [[True, True, False], [True, False, True], [False, True, True]],
        dtype=bool,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 3: 2D float64 array, broadcasted where, axis=[0], keepdims=False
    a = np.array([[-1.5, 2.3, 4.1], [0.5, -3.2, 1.1]], dtype=np.float64)
    axis = [0]
    keepdims = False
    initial = np.array(np.inf, dtype=np.float64)
    where = np.array([[True, False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 4: 3D int64 array, multiple reduction axes, axis=[0, 2], keepdims=True
    a = np.arange(-12, 12, dtype=np.int64).reshape(2, 3, 4)
    axis = [0, 2]
    keepdims = True
    initial = np.array(500, dtype=np.int64)
    where = a > -5
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 5: 3D float32 array, axis=[1], keepdims=False
    a = np.random.uniform(-10.0, 10.0, size=(3, 4, 2)).astype(np.float32)
    axis = [1]
    keepdims = False
    initial = np.array(20.0, dtype=np.float32)
    where = np.ones((3, 4, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 6: 4D float32 array, axis=[1, 3], keepdims=False
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = [1, 3]
    keepdims = False
    initial = np.array(100.0, dtype=np.float32)
    where = np.random.choice([True, False], size=(2, 3, 2, 4))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 7: 1D int16 array, negative axis=[-1], keepdims=True
    a = np.array([-100, 250, -3200, 1500, 0], dtype=np.int16)
    axis = [-1]
    keepdims = True
    initial = np.array(10000, dtype=np.int16)
    where = np.array([True, True, False, True, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 8: 2D uint8 array, axis=[0, 1], keepdims=False
    a = np.array([[12, 45, 230], [5, 128, 99]], dtype=np.uint8)
    axis = [0, 1]
    keepdims = False
    initial = np.array(255, dtype=np.uint8)
    where = np.array([[False, True, True], [True, False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 9: 3D float64 array, broadcasted where column-wise, axis=[-2], keepdims=True
    a = np.linspace(-5.0, 5.0, 24, dtype=np.float64).reshape(2, 4, 3)
    axis = [-2]
    keepdims = True
    initial = np.array(50.0, dtype=np.float64)
    where = np.array([[[True], [False], [True], [True]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 10: 4D int32 array, reducing all axes=[0, 1, 2, 3], keepdims=True
    a = np.random.randint(-100, 100, size=(2, 2, 3, 2), dtype=np.int32)
    axis = [0, 1, 2, 3]
    keepdims = True
    initial = np.array(200, dtype=np.int32)
    where = np.ones((2, 2, 3, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 11: 5D float32 array, axis=[2, 4], keepdims=False
    a = np.random.uniform(-1.0, 1.0, size=(2, 2, 3, 2, 2)).astype(np.float32)
    axis = [2, 4]
    keepdims = False
    initial = np.array(10.0, dtype=np.float32)
    where = a > 0.0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.min_3"] = min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.min_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.min_3'.")


check_valid('jax.numpy.min', generated_inputs['jax.numpy.min_3'], lib="jax", suffix=3)
