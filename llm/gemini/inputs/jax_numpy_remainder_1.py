
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def remainder_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays with mixed signs
    x1 = np.array([3, -1, 4, -8], dtype=np.int32)
    x2 = np.array([2, 3, -5, -3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float32 arrays with broadcasting
    x1 = np.array([[3.5, -1.2, 4.8], [8.1, 5.5, -2.9]], dtype=np.float32)
    x2 = np.array([[2.0, 3.0, -1.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 0D scalar arrays
    x1 = np.array(17.5, dtype=np.float64)
    x2 = np.array(4.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 3D float64 arrays of same shape
    x1 = np.array(
        [[[10.0, -15.0], [20.0, 25.0]], [[-30.0, 35.0], [40.0, -45.0]]],
        dtype=np.float64,
    )
    x2 = np.array(
        [[[3.0, 4.0], [-6.0, 7.0]], [[8.0, -9.0], [11.0, 12.0]]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 1D int32 arrays
    x1 = np.array([10, 25, 50, 100], dtype=np.int32)
    x2 = np.array([3, 4, 7, 9], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 2D int64 arrays
    x1 = np.array([[100, -200], [300, -400]], dtype=np.int64)
    x2 = np.array([[-30, 45], [-70, 80]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D arrays with complex broadcasting
    x1 = np.arange(1, 7, dtype=np.float32).reshape(2, 1, 3, 1)
    x2 = np.arange(2, 12, dtype=np.float32).reshape(1, 2, 1, 5)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D float32 arrays with fractional divisors
    x1 = np.array([0.5, 1.25, 2.75, -3.1], dtype=np.float32)
    x2 = np.array([0.25, 0.5, 1.0, -0.75], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D array and 1D array broadcasting across columns
    x1 = np.array([[5, 10, 15], [20, 25, 30]], dtype=np.int32)
    x2 = np.array([4, 6, 8], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D array and column vector broadcasting
    x1 = np.array([[7.0, 14.0, 21.0], [28.0, 35.0, 42.0]], dtype=np.float64)
    x2 = np.array([[5.0], [9.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.remainder_1"] = remainder_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.remainder_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.remainder_1'.")


check_valid('jax.numpy.remainder', generated_inputs['jax.numpy.remainder_1'], lib="jax", suffix=1)
