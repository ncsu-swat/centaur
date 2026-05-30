
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ediff1d_inputs():
    list_of_inputs = []

    # 1. 1D int array with positive elements
    ary = np.array([2, 3, 5, 9, 1, 4], dtype=np.int32)
    input_dict = {"ary": ary, "to_end": 20, "to_begin": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. 1D float array
    ary = np.array([1.5, 2.5, 4.0, 7.5], dtype=np.float32)
    input_dict = {"ary": ary, "to_end": 5, "to_begin": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. 2D int array
    ary = np.array([[2, -1, 4, 7], [3, 5, -6, 9]], dtype=np.int32)
    input_dict = {"ary": ary, "to_end": -1, "to_begin": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. 3D float array
    ary = np.random.randn(2, 3, 2).astype(np.float32)
    input_dict = {"ary": ary, "to_end": 10, "to_begin": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. Small 1D array of size 2
    ary = np.array([10, 20], dtype=np.int32)
    input_dict = {"ary": ary, "to_end": 1, "to_begin": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. 1D array with negative values
    ary = np.array([-5, -10, -20, -50], dtype=np.int32)
    input_dict = {"ary": ary, "to_end": -5, "to_begin": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. Larger 1D range array
    ary = np.arange(100, dtype=np.int32)
    input_dict = {"ary": ary, "to_end": 99, "to_begin": 100}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. 1D float64 array
    ary = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
    input_dict = {"ary": ary, "to_end": 1, "to_begin": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. 1D int16 array
    ary = np.array([100, 200, 300, 400], dtype=np.int16)
    input_dict = {"ary": ary, "to_end": 5, "to_begin": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. 2D float64 array
    ary = np.random.randn(5, 5).astype(np.float64)
    input_dict = {"ary": ary, "to_end": 0, "to_begin": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ediff1d_2"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_2'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_2'], lib="jax", suffix=2)
