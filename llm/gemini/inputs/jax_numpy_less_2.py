
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def less_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive float
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = 2.5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 2: 2D float32 array with negative values, negative float
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32)
    y = -2.5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 3: 3D int32 array, float representing integer
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    y = 5.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 4: 1D float64 array, zero float
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float64)
    y = 0.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 5: 4D float32 array, positive float
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    y = 0.5
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 6: 2D int16 array, float
    x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16)
    y = 35.2
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 7: 1D float32 array with inf and nan (though nan comparisons are False), float
    x = np.array([-np.inf, np.inf, np.nan, 0.0], dtype=np.float32)
    y = 1.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 8: Large 2D float32 array, float
    x = np.random.uniform(-100.0, 100.0, size=(50, 50)).astype(np.float32)
    y = 10.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 9: 0D array (scalar array) of float32, float
    x = np.array(5.5, dtype=np.float32)
    y = 6.0
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    # Input 10: 5D float32 array, negative float
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    y = -1.2
    list_of_inputs.append({"x": copy.deepcopy(x), "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.less_2"] = less_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.less_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.less_2'.")


check_valid('jax.numpy.less', generated_inputs['jax.numpy.less_2'], lib="jax", suffix=2)
