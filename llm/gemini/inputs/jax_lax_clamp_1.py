
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def clamp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of the same shape
    min_val = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
    x = np.array([-2.0, 0.2, 1.5], dtype=np.float32)
    max_val = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 2: 2D float32 arrays of the same shape with negative values
    min_val = np.full((3, 3), -2.0, dtype=np.float32)
    x = np.random.randn(3, 3).astype(np.float32) * 5.0
    max_val = np.full((3, 3), 2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 3: 0D scalar float32 arrays
    min_val = np.array(-5.0, dtype=np.float32)
    x = np.array(3.14, dtype=np.float32)
    max_val = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 4: 3D float64 arrays of the same shape
    min_val = np.zeros((2, 3, 4), dtype=np.float64)
    x = np.random.randn(2, 3, 4).astype(np.float64)
    max_val = np.ones((2, 3, 4), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 5: 1D int32 arrays of the same shape
    min_val = np.array([0, 10, 20], dtype=np.int32)
    x = np.array([-5, 15, 35], dtype=np.int32)
    max_val = np.array([5, 20, 30], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 6: 2D int64 arrays of the same shape
    min_val = np.full((4, 2), -100, dtype=np.int64)
    x = np.random.randint(-200, 200, size=(4, 2), dtype=np.int64)
    max_val = np.full((4, 2), 100, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 7: 0D scalar min and max broadcasting over 2D float32 x
    min_val = np.array(-1.0, dtype=np.float32)
    x = np.random.randn(5, 5).astype(np.float32)
    max_val = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 8: 0D scalar min and max broadcasting over 3D float32 x
    min_val = np.array(0.0, dtype=np.float32)
    x = np.random.randn(3, 4, 2).astype(np.float32)
    max_val = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 9: 4D float32 arrays of the same shape
    min_val = np.full((2, 2, 3, 3), -0.5, dtype=np.float32)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    max_val = np.full((2, 2, 3, 3), 0.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 10: 2D float64 arrays of the same shape
    min_val = np.full((2, 5), -3.0, dtype=np.float64)
    x = np.random.randn(2, 5).astype(np.float64) * 4.0
    max_val = np.full((2, 5), 3.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 11: 0D scalar min and max broadcasting over 1D int32 x
    min_val = np.array(0, dtype=np.int32)
    x = np.array([-10, 5, 20], dtype=np.int32)
    max_val = np.array(10, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    # Input 12: 0D scalar min and max broadcasting over 2D int64 x
    min_val = np.array(-50, dtype=np.int64)
    x = np.random.randint(-100, 100, size=(3, 3), dtype=np.int64)
    max_val = np.array(50, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"min": min_val, "x": x, "max": max_val})
    )

    return list_of_inputs


generated_inputs["jax.lax.clamp_1"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clamp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clamp_1'.")


check_valid('jax.lax.clamp', generated_inputs['jax.lax.clamp_1'], lib="jax", suffix=1)
