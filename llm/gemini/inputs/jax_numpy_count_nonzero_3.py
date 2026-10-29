
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    a = np.array([0, 1, 0, -3, 5, 0], dtype=np.int32)
    input_dict = {"a": a, "axis": (0,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array with keepdims=True
    a = np.array([[0.0, 1.5, 0.0], [-2.3, 0.0, 4.1]], dtype=np.float32)
    input_dict = {"a": a, "axis": (1,), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array counting along axis 0
    a = np.array([[1, 0, 2], [0, 0, -4], [5, 6, 0]], dtype=np.int64)
    input_dict = {"a": a, "axis": (0,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D boolean array with multiple axes
    a = np.array(
        [[[True, False], [False, False]], [[True, True], [False, True]]],
        dtype=bool,
    )
    input_dict = {"a": a, "axis": (0, 2), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float array with negative axis
    a = np.random.randn(2, 3, 4).astype(np.float64)
    a[a < 0] = 0.0
    input_dict = {"a": a, "axis": (-1,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float array with multiple reduction axes
    a = np.zeros((2, 3, 4, 5), dtype=np.float32)
    a[0, 1, 2, 3] = 1.0
    a[1, 2, 0, 4] = -2.5
    input_dict = {"a": a, "axis": (1, 2), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex array
    a = np.array(
        [[0 + 0j, 1 + 2j, 0 + 0j], [-1 + 0j, 0 + 3j, 0 + 0j]],
        dtype=np.complex64,
    )
    input_dict = {"a": a, "axis": (0,), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float array with negative axis index
    a = np.array([-1.0, 0.0, 2.0, -3.0, 0.0], dtype=np.float32)
    input_dict = {"a": a, "axis": (-1,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D integer array reducing outer dimensions
    a = np.random.choice([0, 1, -1, 2], size=(3, 2, 2, 4)).astype(np.int32)
    input_dict = {"a": a, "axis": (0, 3), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D integer array reducing along middle axis
    a = np.array(
        [[[0, 1], [-2, 0], [3, 0]], [[0, 0], [1, -1], [0, 2]]], dtype=np.int16
    )
    input_dict = {"a": a, "axis": (1,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.count_nonzero_3"] = count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.count_nonzero_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.count_nonzero_3'.")


check_valid('jax.numpy.count_nonzero', generated_inputs['jax.numpy.count_nonzero_3'], lib="jax", suffix=3)
