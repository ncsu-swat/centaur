
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def clip_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, 0D scalar bounds
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    min_val = np.array(-0.5, dtype=np.float32)
    max_val = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 2: 2D float32 array, 0D scalar bounds
    arr = np.random.randn(4, 5).astype(np.float32)
    min_val = np.array(-1.0, dtype=np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 3: 1D int32 array, 0D int32 bounds
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    min_val = np.array(2, dtype=np.int32)
    max_val = np.array(5, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 4: 2D float64 array, broadcastable 1D bounds along columns
    arr = np.linspace(-10.0, 10.0, 12, dtype=np.float64).reshape(3, 4)
    min_val = np.array([-5.0, -4.0, -3.0, -2.0], dtype=np.float64)
    max_val = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 5: 3D float32 array, matching shape bounds
    arr = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    min_val = np.full((2, 3, 4), -2.0, dtype=np.float32)
    max_val = np.full((2, 3, 4), 2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 6: 1D int64 array with negative values, 0D bounds
    arr = np.array([-100, -50, 0, 50, 100], dtype=np.int64)
    min_val = np.array(-25, dtype=np.int64)
    max_val = np.array(75, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 7: 4D float32 array, 0D bounds
    arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    min_val = np.array(-0.1, dtype=np.float32)
    max_val = np.array(0.1, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 8: 2D int16 array, broadcastable bounds along rows
    arr = np.arange(12, dtype=np.int16).reshape(4, 3)
    min_val = np.array([[2], [4], [6], [8]], dtype=np.int16)
    max_val = np.array([[5], [7], [9], [11]], dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 9: 1D float32 array with min > max (inverted range)
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    min_val = np.array(5.0, dtype=np.float32)
    max_val = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 10: 0D scalar float32 array, 0D bounds
    arr = np.array(3.5, dtype=np.float32)
    min_val = np.array(0.0, dtype=np.float32)
    max_val = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    # Input 11: 2D uint8 array with byte-range bounds
    arr = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
    min_val = np.array(50, dtype=np.uint8)
    max_val = np.array(200, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "min": min_val, "max": max_val})
    )

    return list_of_inputs


generated_inputs["jax.numpy.clip_1"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.clip_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.clip_1'.")


check_valid('jax.numpy.clip', generated_inputs['jax.numpy.clip_1'], lib="jax", suffix=1)
