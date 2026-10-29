
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def where_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean condition, 1D float32 arrays of same shape
    condition = np.array([True, False, True, False, True], dtype=np.bool_)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 2: 2D arrays with integers (including negative values)
    condition = np.array(
        [[True, False], [False, True], [True, True]], dtype=np.bool_
    )
    x = np.array([[10, -20], [30, -40], [50, 60]], dtype=np.int32)
    y = np.array([[-1, 2], [-3, 4], [-5, -6]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 3: 0D scalar arrays
    condition = np.array(True, dtype=np.bool_)
    x = np.array(42.5, dtype=np.float64)
    y = np.array(-10.2, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 4: Broadcasting condition (1D) across 2D arrays
    condition = np.array([True, False, True], dtype=np.bool_)
    x = np.ones((4, 3), dtype=np.float32) * 5.0
    y = np.zeros((4, 3), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 5: Broadcasting x and y with a 2D condition
    condition = np.array([[True, False, True], [False, True, False]])
    x = np.array([[1.0]], dtype=np.float64)
    y = np.array([[-1.0, -2.0, -3.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 6: 3D tensors with float32
    condition = np.random.rand(2, 3, 4) > 0.5
    x = np.random.randn(2, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 7: 4D tensors with int64
    condition = np.random.choice([True, False], size=(2, 2, 2, 2))
    x = np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int64)
    y = np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 8: Complex numbers
    condition = np.array([True, False, True], dtype=np.bool_)
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j], dtype=np.complex64)
    y = np.array([0.0 - 1.0j, 2.0 + 0.0j, -4.0 - 5.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 9: Mixed typecasting (int32 and float32)
    condition = np.array([[True, False], [True, True]], dtype=np.bool_)
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 10: Special values (inf, -inf, nan)
    condition = np.array([True, False, True, False], dtype=np.bool_)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0, -np.inf], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    # Input 11: Boolean x and y (logic operation via where)
    condition = np.array([True, False, True, False], dtype=np.bool_)
    x = np.array([False, False, True, True], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"condition": condition, "x": x, "y": y})
    )

    return list_of_inputs


generated_inputs["jax.numpy.where_1"] = where_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.where_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.where_1'.")


check_valid('jax.numpy.where', generated_inputs['jax.numpy.where_1'], lib="jax", suffix=1)
