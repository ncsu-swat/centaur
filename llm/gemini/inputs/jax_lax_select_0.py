
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def select_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32
    pred = np.array([True, False, True, False], dtype=bool)
    on_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    on_false = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 2: 2D arrays, int32
    pred = np.random.choice([True, False], size=(3, 3))
    on_true = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    on_false = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 3: 0D arrays (scalars), float64
    pred = np.array(True, dtype=bool)
    on_true = np.array(3.14, dtype=np.float64)
    on_false = np.array(-3.14, dtype=np.float64)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 4: 3D arrays, int16
    pred = np.random.choice([True, False], size=(2, 2, 2))
    on_true = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int16)
    on_false = np.random.randint(-100, 100, size=(2, 2, 2)).astype(np.int16)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 5: 4D arrays, uint8
    pred = np.random.choice([True, False], size=(1, 2, 2, 1))
    on_true = np.random.randint(0, 255, size=(1, 2, 2, 1)).astype(np.uint8)
    on_false = np.random.randint(0, 255, size=(1, 2, 2, 1)).astype(np.uint8)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 6: 1D arrays, float16 with negative values
    pred = np.array([False, False, True], dtype=bool)
    on_true = np.array([-0.5, 0.0, 0.5], dtype=np.float16)
    on_false = np.array([-1.5, -2.5, -3.5], dtype=np.float16)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 7: 2D arrays, complex64
    pred = np.random.choice([True, False], size=(2, 3))
    on_true = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    on_false = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 8: 3D arrays, int64 with negative values
    pred = np.random.choice([True, False], size=(2, 1, 3))
    on_true = np.random.randint(-1000, 1000, size=(2, 1, 3)).astype(np.int64)
    on_false = np.random.randint(-1000, 1000, size=(2, 1, 3)).astype(np.int64)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 9: 1D arrays, bool for on_true and on_false
    pred = np.array([True, True, False, False], dtype=bool)
    on_true = np.array([True, False, True, False], dtype=bool)
    on_false = np.array([False, True, False, True], dtype=bool)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    # Input 10: 5D arrays, float32
    pred = np.random.choice([True, False], size=(1, 1, 2, 2, 2))
    on_true = np.random.randn(1, 1, 2, 2, 2).astype(np.float32)
    on_false = np.random.randn(1, 1, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"pred": pred, "on_true": on_true, "on_false": on_false})

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.lax.select"] = select_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.select'.")


check_valid('jax.lax.select', generated_inputs['jax.lax.select'], lib="jax", suffix=0)
