
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def diff_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, first-order diff along axis 0
    a = np.array([1, 2, 4, 7, 0], dtype=np.int32)
    n = 1
    axis = 0
    prepend = np.array([0], dtype=np.int32)
    append = np.array([5], dtype=np.int32)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array, second-order diff with negative values
    a = np.array([-1.5, 2.0, 0.5, 3.2, -4.1], dtype=np.float32)
    n = 2
    axis = -1
    prepend = np.array([1.0, 2.0], dtype=np.float32)
    append = np.array([0.0], dtype=np.float32)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, diff along axis 0
    a = np.zeros((3, 4), dtype=np.float64)
    n = 1
    axis = 0
    prepend = np.ones((1, 4), dtype=np.float64)
    append = np.full((2, 4), 5.0, dtype=np.float64)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array, higher order difference (n=3) along axis 1
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    n = 3
    axis = 1
    prepend = np.zeros((3, 2), dtype=np.int64)
    append = np.ones((3, 2), dtype=np.int64)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, diff along axis 2
    a = np.random.randn(2, 3, 5).astype(np.float32)
    n = 1
    axis = 2
    prepend = np.random.randn(2, 3, 1).astype(np.float32)
    append = np.random.randn(2, 3, 1).astype(np.float32)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, 0-th order difference (n=0) along axis 1
    a = np.random.randn(2, 4, 3).astype(np.float32)
    n = 0
    axis = 1
    prepend = np.random.randn(2, 1, 3).astype(np.float32)
    append = np.random.randn(2, 2, 3).astype(np.float32)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D int16 array with negative axis specification
    a = np.random.randint(-10, 10, size=(2, 2, 3, 4), dtype=np.int16)
    n = 1
    axis = -2
    prepend = np.zeros((2, 2, 1, 4), dtype=np.int16)
    append = np.zeros((2, 2, 1, 4), dtype=np.int16)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D uint8 array
    a = np.array([10, 20, 30, 40], dtype=np.uint8)
    n = 1
    axis = 0
    prepend = np.array([5], dtype=np.uint8)
    append = np.array([50], dtype=np.uint8)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex128 array along axis 0
    a = np.array([[1 + 2j, 2 - 1j], [3 + 0j, -1 - 1j]], dtype=np.complex128)
    n = 1
    axis = 0
    prepend = np.array([[0 + 0j, 0 + 0j]], dtype=np.complex128)
    append = np.array([[1 + 1j, 1 - 1j]], dtype=np.complex128)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array with negative values, n=2 along axis -1
    a = np.array([[-10, 20, -30], [5, -15, 25]], dtype=np.int32)
    n = 2
    axis = -1
    prepend = np.array([[0], [0]], dtype=np.int32)
    append = np.array([[10], [10]], dtype=np.int32)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float64 array, n=2 along axis 0
    a = np.ones((4, 2, 2), dtype=np.float64)
    n = 2
    axis = 0
    prepend = np.zeros((1, 2, 2), dtype=np.float64)
    append = np.zeros((1, 2, 2), dtype=np.float64)
    input_dict = {"a": a, "n": n, "axis": axis, "prepend": prepend, "append": append}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.diff"] = diff_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diff' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diff'.")


check_valid('jax.numpy.diff', generated_inputs['jax.numpy.diff'], lib="jax", suffix=0)
