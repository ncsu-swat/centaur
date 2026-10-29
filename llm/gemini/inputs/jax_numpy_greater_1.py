
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def greater_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with float32 values (including negatives)
    x = np.array([5.0, 9.0, -2.0], dtype=np.float32)
    y = np.array([4.0, -1.0, 6.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D arrays with broadcasting (2x3 and 1x3)
    x = np.array([[5, -6, 7], [-2, 5, 9]], dtype=np.int32)
    y = np.array([[-4, 3, 10]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: Scalar-shaped (0D) arrays
    x = np.array(5, dtype=np.int64)
    y = np.array(2, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D arrays with float64 values and same shape
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Broadcast across multiple dimensions (3D and 1D)
    x = np.ones((2, 4, 3), dtype=np.float32)
    y = np.zeros((3,), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: Unsigned integer types
    x = np.array([10, 20, 30], dtype=np.uint8)
    y = np.array([15, 20, 25], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Boolean arrays
    x = np.array([[True, False], [True, True]], dtype=bool)
    y = np.array([[False, False], [True, False]], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D tensors broadcasting (1, 3, 1, 5) with (2, 1, 4, 1)
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    y = np.random.randn(2, 1, 4, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Floating point special values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    y = np.array([1000.0, -1000.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Empty arrays (0-length dimension)
    x = np.empty((0, 5), dtype=np.float32)
    y = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: int16 type with negative values
    x = np.array([-32768, -100, 0, 100, 32767], dtype=np.int16)
    y = np.array([0, -100, -1, 99, 32766], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.greater_1"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_1'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_1'], lib="jax", suffix=1)
