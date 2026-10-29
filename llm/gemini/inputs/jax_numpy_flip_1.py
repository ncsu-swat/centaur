
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def flip_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    m = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 2: 1D int32 array, negative axis -1
    m = np.array([-10, 0, 10, 20, -30], dtype=np.int32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 3: 2D float64 array, axis 0
    m = np.random.randn(3, 4).astype(np.float64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 4: 2D int64 array, axis 1
    m = np.arange(12, dtype=np.int64).reshape((4, 3))
    axis = 1
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 5: 2D boolean array, negative axis -2
    m = np.array([[True, False], [False, True]], dtype=np.bool_)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 6: 3D float32 array, axis 2
    m = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 7: 3D complex64 array, axis -1
    m = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(
        np.complex64
    )
    axis = -1
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 8: 3D uint8 array, axis 1
    m = np.random.randint(0, 255, size=(3, 5, 2), dtype=np.uint8)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 9: 4D float32 array, axis 3
    m = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = 3
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 10: 4D int16 array, negative axis -3
    m = np.random.randint(-100, 100, size=(2, 4, 3, 2), dtype=np.int16)
    axis = -3
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 11: 5D float64 array, axis 4
    m = np.random.randn(1, 2, 2, 3, 4).astype(np.float64)
    axis = 4
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.flip_1"] = flip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flip_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flip_1'.")


check_valid('jax.numpy.flip', generated_inputs['jax.numpy.flip_1'], lib="jax", suffix=1)
