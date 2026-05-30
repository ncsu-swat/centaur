
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dsplit_inputs():
    list_of_inputs = []

    # Input 1: Float32, 3D array, single split index
    ary = np.random.randn(2, 3, 4).astype(np.float32)
    indices_or_sections = [2]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 2: Float32, 3D array, multiple split indices
    ary = np.random.randn(1, 2, 10).astype(np.float32)
    indices_or_sections = [3, 7]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 3: Int32, 3D array, multiple split indices
    ary = np.random.randint(-10, 10, size=(3, 3, 6)).astype(np.int32)
    indices_or_sections = [2, 4]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 4: Float64, 4D array, single split index
    ary = np.ones((2, 2, 8, 2), dtype=np.float64)
    indices_or_sections = [4]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 5: Float32, 3D array, splitting into 5 parts
    ary = np.random.randn(4, 2, 5).astype(np.float32)
    indices_or_sections = [1, 2, 3, 4]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 6: Float32, 3D array, large size, multiple split indices
    ary = np.zeros((3, 1, 12)).astype(np.float32)
    indices_or_sections = [3, 6, 9]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 7: Float16, 3D array, minimal split
    ary = np.random.randn(2, 2, 2).astype(np.float16)
    indices_or_sections = [1]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 8: Float32, 4D array, multiple split indices
    ary = np.random.randn(1, 5, 20, 3).astype(np.float32)
    indices_or_sections = [5, 10, 15]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 9: Complex64, 3D array
    ary = np.random.randn(3, 3, 3).astype(np.complex64)
    indices_or_sections = [1, 2]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    # Input 10: Uint8, 3D array
    ary = np.random.randint(0, 100, size=(5, 5, 5)).astype(np.uint8)
    indices_or_sections = [2, 3]
    list_of_inputs.append({"ary": copy.deepcopy(ary), "indices_or_sections": indices_or_sections})

    return list_of_inputs

generated_inputs["jax.numpy.dsplit_2"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dsplit_2'.")


check_valid('jax.numpy.dsplit', generated_inputs['jax.numpy.dsplit_2'], lib="jax", suffix=2)
