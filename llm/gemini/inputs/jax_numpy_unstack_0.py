
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unstack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis=0
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": 0}))

    # Input 2: 1D int32 array, negative axis=-1
    x = np.array([10, 20, 30], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": -1}))

    # Input 3: 2D float64 array, axis=0
    x = np.arange(6, dtype=np.float64).reshape((2, 3))
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": 0}))

    # Input 4: 2D boolean array, axis=1
    x = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": 1}))

    # Input 5: 2D int64 array, negative axis=-2
    x = np.ones((4, 2), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": -2}))

    # Input 6: 3D float32 array, axis=2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": 2}))

    # Input 7: 3D complex64 array, negative axis=-1
    x = (np.random.randn(3, 2, 2) + 1j * np.random.randn(3, 2, 2)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": -1}))

    # Input 8: 4D uint8 array, axis=1
    x = np.arange(24, dtype=np.uint8).reshape((2, 2, 3, 2))
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": 1}))

    # Input 9: 4D float64 array, negative axis=-3
    x = np.random.randn(1, 3, 2, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": -3}))

    # Input 10: 5D int16 array, negative axis=-5
    x = np.zeros((2, 1, 3, 1, 2), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": -5}))

    # Input 11: 1D single-element array, axis=0
    x = np.array([42.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": 0}))

    return list_of_inputs


generated_inputs["jax.numpy.unstack"] = unstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unstack'.")


check_valid('jax.numpy.unstack', generated_inputs['jax.numpy.unstack'], lib="jax", suffix=0)
