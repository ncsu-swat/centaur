
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split at index 3
    ary = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    indices_or_sections = [3]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 2: 1D array, split at indices 2, 5, 8
    ary = np.arange(10, dtype=np.float32)
    indices_or_sections = [2, 5, 8]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 3: 2D array, float32, split at index 4
    ary = np.random.randn(4, 8).astype(np.float32)
    indices_or_sections = [4]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 4: 2D array, int32, split at indices 3 and 6
    ary = np.random.randint(-10, 10, size=(3, 9)).astype(np.int32)
    indices_or_sections = [3, 6]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 5: 3D array, float64, split along axis 1 at indices 2 and 4
    ary = np.random.randn(2, 6, 4).astype(np.float64)
    indices_or_sections = [2, 4]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 6: 4D array, float32, split along axis 1 at index 5
    ary = np.ones((2, 10, 2, 2), dtype=np.float32)
    indices_or_sections = [5]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 7: 2D array, bool type, split at index 2
    ary = np.random.choice([True, False], size=(3, 4))
    indices_or_sections = [2]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 8: 1D array with negative values, split at index 2
    ary = np.array([-1.5, -2.5, -3.5, -4.5, -5.5], dtype=np.float32)
    indices_or_sections = [2]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 9: 2D array, split at index 1
    ary = np.zeros((2, 2), dtype=np.float32)
    indices_or_sections = [1]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Input 10: 3D array, split at indices 1, 2, 3 along axis 1
    ary = np.arange(24).reshape(2, 4, 3).astype(np.int64)
    indices_or_sections = [1, 2, 3]
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    return list_of_inputs

generated_inputs["jax.numpy.hsplit_2"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hsplit_2'.")


check_valid('jax.numpy.hsplit', generated_inputs['jax.numpy.hsplit_2'], lib="jax", suffix=2)
