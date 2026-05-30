
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import warnings

def nanvar_inputs():
    list_of_inputs = []

    # Helper to generate random array with NaNs
    def gen_array_with_nans(shape, low=-10.0, high=10.0, dtype=np.float32):
        arr = np.random.uniform(low, high, size=shape).astype(dtype)
        mask = np.random.rand(*shape) < 0.2
        arr[mask] = np.nan
        if np.isnan(arr).all():
            arr.flat[0] = 1.0
        return arr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # 1. 2D array, float32, axis=(0,), keepdims=False
        a = gen_array_with_nans((4, 4), dtype=np.float32)
        axis = (0,)
        where = np.random.rand(4, 4) > 0.1
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float32),
            "ddof": 0,
            "keepdims": False,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 2. 2D array, float64, axis=(1,), keepdims=True, ddof=1
        a = gen_array_with_nans((3, 5), dtype=np.float64)
        axis = (1,)
        where = np.random.rand(3, 5) > 0.1
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float64),
            "ddof": 1,
            "keepdims": True,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 3. 3D array, float32, axis=(0, 2), keepdims=True, ddof=0
        a = gen_array_with_nans((2, 3, 4), dtype=np.float32)
        axis = (0, 2)
        where = np.random.rand(2, 3, 4) > 0.2
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float32),
            "ddof": 0,
            "keepdims": True,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 4. 1D array, float32, axis=(0,), keepdims=False, ddof=1
        a = gen_array_with_nans((10,), dtype=np.float32)
        axis = (0,)
        where = np.random.rand(10) > 0.1
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float32),
            "ddof": 1,
            "keepdims": False,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 5. 4D array, float64, axis=(1, 2), keepdims=False, ddof=2
        a = gen_array_with_nans((2, 2, 3, 3), dtype=np.float64)
        axis = (1, 2)
        where = np.random.rand(2, 2, 3, 3) > 0.1
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float64),
            "ddof": 2,
            "keepdims": False,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 6. 2D array (5, 5), positive, float32, axis=(0,), keepdims=True, ddof=1
        a = gen_array_with_nans((5, 5), low=1.0, high=100.0, dtype=np.float32)
        axis = (0,)
        where = np.random.rand(5, 5) > 0.05
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float32),
            "ddof": 1,
            "keepdims": True,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 7. 3D array (3, 2, 2), negative, float32, axis=(2,), keepdims=True, ddof=0
        a = gen_array_with_nans((3, 2, 2), low=-50.0, high=-1.0, dtype=np.float32)
        axis = (2,)
        where = np.random.rand(3, 2, 2) > 0.1
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float32),
            "ddof": 0,
            "keepdims": True,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 8. 2D array (2, 6), large, float64, axis=(1,), keepdims=False, ddof=1
        a = gen_array_with_nans((2, 6), low=1000.0, high=5000.0, dtype=np.float64)
        axis = (1,)
        where = np.random.rand(2, 6) > 0.1
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float64),
            "ddof": 1,
            "keepdims": False,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 9. 4D array (2, 1, 4, 2), float32, axis=(0, 2, 3), keepdims=True, ddof=1
        a = gen_array_with_nans((2, 1, 4, 2), dtype=np.float32)
        axis = (0, 2, 3)
        where = np.random.rand(2, 1, 4, 2) > 0.2
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float32),
            "ddof": 1,
            "keepdims": True,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

        # 10. 2D array (8, 8), float32, axis=(0, 1), keepdims=False, ddof=0, all-True where
        a = gen_array_with_nans((8, 8), dtype=np.float32)
        axis = (0, 1)
        where = np.ones((8, 8), dtype=bool)
        masked_a = np.where(where, a, np.nan)
        mean = np.nanmean(masked_a, axis=axis, keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        input_dict = {
            "a": a,
            "axis": axis,
            "dtype": np.dtype(np.float32),
            "ddof": 0,
            "keepdims": False,
            "where": where,
            "mean": mean
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nanvar_2"] = nanvar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanvar_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanvar_2'.")


check_valid('jax.numpy.nanvar', generated_inputs['jax.numpy.nanvar_2'], lib="jax", suffix=2)
