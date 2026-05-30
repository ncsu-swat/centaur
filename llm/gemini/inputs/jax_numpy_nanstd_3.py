
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nanstd_inputs():
    list_of_inputs = []

    def make_input(a, axis, dtype, ddof, keepdims, where):
        # Apply mask to safely compute mean
        a_masked = np.where(where, a, np.nan)
        mean = np.nanmean(a_masked, axis=axis, keepdims=True)
        # Fill any remaining NaNs in mean to ensure stable computation
        mean = np.where(np.isnan(mean), 0.0, mean)
        return {
            "a": a.astype(dtype),
            "axis": axis,
            "dtype": dtype,
            "ddof": ddof,
            "keepdims": keepdims,
            "where": where,
            "mean": mean.astype(dtype)
        }

    # Input 1
    a = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(make_input(a, (0,), np.float32, 0, True, where))

    # Input 2
    a = np.array([[2.0, 4.0, np.nan], [np.nan, 1.0, 5.0]])
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(make_input(a, (1,), np.float32, 1, False, where))

    # Input 3
    a = np.random.randn(2, 3, 4)
    a[0, 1, 2] = np.nan
    where = np.random.choice([True, False], size=a.shape, p=[0.9, 0.1])
    list_of_inputs.append(make_input(a, (0, 2), np.float64, 0, True, where))

    # Input 4
    a = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(make_input(a, (0,), np.float32, 1, False, where))

    # Input 5
    a = np.random.randn(3, 3, 3)
    a[1, 1, 1] = np.nan
    where = np.random.choice([True, False], size=a.shape, p=[0.8, 0.2])
    list_of_inputs.append(make_input(a, (1,), np.float32, 2, True, where))

    # Input 6
    a = np.array([[np.nan, 2.0], [3.0, np.nan]])
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(make_input(a, (0, 1), np.float64, 0, False, where))

    # Input 7
    a = np.random.randn(2, 2, 2, 2)
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(make_input(a, (1, 3), np.float32, 1, True, where))

    # Input 8
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    where = np.array([[True, False, True], [False, True, True]], dtype=bool)
    list_of_inputs.append(make_input(a, (0,), np.float32, 0, True, where))

    # Input 9
    a = np.random.randn(3, 4, 2)
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(make_input(a, (2,), np.float64, 1, False, where))

    # Input 10
    a = np.random.randn(5, 2)
    where = np.ones_like(a, dtype=bool)
    list_of_inputs.append(make_input(a, (1,), np.float32, 2, True, where))

    return list_of_inputs

generated_inputs["jax.numpy.nanstd_3"] = nanstd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanstd_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanstd_3'.")


check_valid('jax.numpy.nanstd', generated_inputs['jax.numpy.nanstd_3'], lib="jax", suffix=3)
