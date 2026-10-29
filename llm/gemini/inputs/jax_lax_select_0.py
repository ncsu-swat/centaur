
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_select_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar boolean and float32
    pred = np.array(True, dtype=np.bool_)
    on_true = np.array(1.5, dtype=np.float32)
    on_false = np.array(-2.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 2: 1D arrays with int32
    pred = np.array([True, False, True, False, True], dtype=np.bool_)
    on_true = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    on_false = np.array([-1, -2, -3, -4, -5], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 3: 2D arrays with float32
    pred = np.array([[True, False], [False, True]], dtype=np.bool_)
    on_true = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    on_false = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 4: 2D arrays with float64
    pred = np.random.choice([True, False], size=(3, 4)).astype(np.bool_)
    on_true = np.random.randn(3, 4).astype(np.float64)
    on_false = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 5: 3D arrays with int64
    pred = np.random.choice([True, False], size=(2, 3, 2)).astype(np.bool_)
    on_true = np.random.randint(-100, 100, size=(2, 3, 2), dtype=np.int64)
    on_false = np.random.randint(-100, 100, size=(2, 3, 2), dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 6: 1D arrays with complex64
    pred = np.array([True, False, False, True], dtype=np.bool_)
    on_true = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j, 0.0 + 0.0j], dtype=np.complex64)
    on_false = np.array([0.0 - 1.0j, 2.0 + 3.0j, -1.0 - 1.0j, 5.0 + 2.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 7: 4D arrays with float32
    pred = np.random.choice([True, False], size=(2, 2, 3, 4)).astype(np.bool_)
    on_true = np.ones((2, 2, 3, 4), dtype=np.float32)
    on_false = np.zeros((2, 2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 8: 2D arrays with uint8
    pred = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)
    on_true = np.array([[255, 128, 64], [32, 16, 8]], dtype=np.uint8)
    on_false = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 9: 1D boolean branches
    pred = np.array([True, False, True], dtype=np.bool_)
    on_true = np.array([False, False, True], dtype=np.bool_)
    on_false = np.array([True, True, False], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 10: 3D arrays with int16
    pred = np.zeros((2, 1, 4), dtype=np.bool_)
    on_true = np.full((2, 1, 4), 100, dtype=np.int16)
    on_false = np.full((2, 1, 4), -100, dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    # Input 11: 0D scalar int32
    pred = np.array(False, dtype=np.bool_)
    on_true = np.array(42, dtype=np.int32)
    on_false = np.array(-42, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"pred": pred, "on_true": on_true, "on_false": on_false})
    )

    return list_of_inputs


generated_inputs["jax.lax.select"] = lax_select_inputs()

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
