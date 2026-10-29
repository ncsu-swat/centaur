
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmax_inputs():
    list_of_inputs = []

    # Input 1: True and 1D float32 array with negative, positive and NaN values
    x1 = True
    x2 = np.array([-2.5, 0.0, 1.5, np.nan, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: False and 2D int32 array
    x1 = False
    x2 = np.array([[-3, 0], [1, 5]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: True and 3D float64 array with infinities
    x1 = True
    x2 = np.array(
        [[[-np.inf, 0.5], [1.0, np.inf]], [[-1.0, np.nan], [2.0, 3.0]]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: False and 1D boolean array
    x1 = False
    x2 = np.array([True, False, False, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: True and 0D scalar float32 array
    x1 = True
    x2 = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: False and 4D float32 array
    x1 = False
    x2 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: True and 2D uint8 array
    x1 = True
    x2 = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: False and 1D float16 array with NaNs and infs
    x1 = False
    x2 = np.array([np.nan, -np.inf, 0.0, np.inf], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: True and 2D int64 array with negative values
    x1 = True
    x2 = np.array([[-10, -5, 0], [1, 2, 10]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: False and 3D int16 array
    x1 = False
    x2 = np.array(
        [[[-1, 0], [2, -3]], [[4, -5], [6, 7]]],
        dtype=np.int16,
    )
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.fmax_7"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_7'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_7'], lib="jax", suffix=7)
