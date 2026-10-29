
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unstack_inputs():
    list_of_inputs = []

    # Input 1: 1D array along axis 0
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array along axis 0
    x = np.arange(10, dtype=np.int32).reshape(2, 5)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array along axis 1
    x = np.random.randn(4, 3).astype(np.float64)
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array with negative axis
    x = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D boolean array along axis 0
    x = np.array(
        [[[True, False], [False, True]], [[True, True], [False, False]]],
        dtype=np.bool_,
    )
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D complex array along axis 1
    x = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(
        np.complex64
    )
    input_dict = {"x": x, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float array along negative axis -2
    x = np.random.uniform(-5.0, 5.0, size=(3, 2, 4)).astype(np.float32)
    input_dict = {"x": x, "axis": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D uint8 array along axis 2
    x = np.random.randint(0, 256, size=(2, 2, 3, 2), dtype=np.uint8)
    input_dict = {"x": x, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float array along axis 3
    x = np.random.randn(1, 2, 1, 4).astype(np.float32)
    input_dict = {"x": x, "axis": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D single-element array along axis 0
    x = np.array([42], dtype=np.int16)
    input_dict = {"x": x, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D array along negative axis -1
    x = np.ones((2, 1, 2, 1, 3), dtype=np.float32)
    input_dict = {"x": x, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.unstack"] = unstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.unstack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.unstack'.")


check_valid('jax.lax.unstack', generated_inputs['jax.lax.unstack'], lib="jax", suffix=0)
