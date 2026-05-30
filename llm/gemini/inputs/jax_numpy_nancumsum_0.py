
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nancumsum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 with NaNs, axis 0
    a = np.array([1.0, np.nan, 3.0, -4.0, np.nan], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 with NaNs, axis 1
    a = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 with NaNs, axis 0
    a = np.array([[np.nan, 2.0], [3.0, np.nan], [-1.0, 4.0]], dtype=np.float64)
    input_dict = {"a": a, "axis": 0, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 with NaNs, axis 2
    a = np.random.randn(2, 3, 4).astype(np.float32)
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    input_dict = {"a": a, "axis": 2, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64, positive & negative values, axis -1
    a = np.array([-10.0, np.nan, 20.0, -30.0], dtype=np.float64)
    input_dict = {"a": a, "axis": -1, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64, axis -2
    a = np.random.randn(5, 5).astype(np.float64)
    a[2, 2] = np.nan
    input_dict = {"a": a, "axis": -2, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32, axis 1, dtype converted to float64
    a = np.random.randn(2, 2, 2).astype(np.float32)
    a[1, 1, 1] = np.nan
    input_dict = {"a": a, "axis": 1, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32, axis 3
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    a[0, 0, 1, 1] = np.nan
    input_dict = {"a": a, "axis": 3, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32, all NaNs
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float16, axis 0
    a = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float16)
    input_dict = {"a": a, "axis": 0, "dtype": np.float16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.nancumsum"] = nancumsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nancumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nancumsum'.")


check_valid('jax.numpy.nancumsum', generated_inputs['jax.numpy.nancumsum'], lib="jax", suffix=0)
