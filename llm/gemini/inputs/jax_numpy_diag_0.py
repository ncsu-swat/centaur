
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def diag_inputs():
    list_of_inputs = []

    # Input 1: 1-D int32 array, default diagonal (k=0)
    v = np.array([1, 2, 3, 4], dtype=np.int32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 2: 1-D float32 array, positive offset (k=1)
    v = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    k = 1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 3: 1-D float64 array, negative offset (k=-2)
    v = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    k = -2
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 4: 2-D square float32 matrix, main diagonal (k=0)
    v = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 5: 2-D rectangular int64 matrix (more columns), positive offset (k=1)
    v = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int64
    )
    k = 1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 6: 2-D rectangular float32 matrix (more rows), negative offset (k=-1)
    v = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    k = -1
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 7: 1-D boolean array, k=0
    v = np.array([True, False, True], dtype=np.bool_)
    k = 0
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 8: 2-D complex64 matrix, positive offset (k=2)
    v = np.array(
        [
            [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j],
            [5 + 6j, 6 + 7j, 7 + 8j, 8 + 9j],
            [9 + 1j, 10 + 2j, 11 + 3j, 12 + 4j],
        ],
        dtype=np.complex64,
    )
    k = 2
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 9: 1-D single-element array, negative offset (k=-3)
    v = np.array([42], dtype=np.int32)
    k = -3
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 10: 2-D square matrix, negative offset (k=-2)
    v = np.arange(16, dtype=np.float32).reshape(4, 4)
    k = -2
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    # Input 11: 1-D uint8 array, positive offset (k=3)
    v = np.array([10, 20, 30], dtype=np.uint8)
    k = 3
    list_of_inputs.append(copy.deepcopy({"v": v, "k": k}))

    return list_of_inputs


generated_inputs["jax.numpy.diag"] = diag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diag'.")


check_valid('jax.numpy.diag', generated_inputs['jax.numpy.diag'], lib="jax", suffix=0)
