
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dsplit_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 array, split into 2 sections
    ary = np.random.randn(3, 1, 4).astype(np.float32)
    indices_or_sections = 2
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 2: 3D float64 array with negative numbers, split into 1 section
    ary = np.random.randn(2, 5, 3).astype(np.float64)
    indices_or_sections = 1
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 3: 4D int32 array, split into 3 sections along axis 2
    ary = np.random.randint(-10, 10, size=(2, 2, 6, 2)).astype(np.int32)
    indices_or_sections = 3
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 4: 3D complex64 array, split into 4 sections
    real_part = np.random.randn(1, 2, 8).astype(np.float32)
    imag_part = np.random.randn(1, 2, 8).astype(np.float32)
    ary = (real_part + 1j * imag_part).astype(np.complex64)
    indices_or_sections = 4
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 5: 5D boolean array, split into 2 sections
    ary = np.random.choice([True, False], size=(2, 1, 2, 3, 4))
    indices_or_sections = 2
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 6: 3D int64 array with large values, split into 5 sections
    ary = np.random.randint(0, 100, size=(4, 4, 10)).astype(np.int64)
    indices_or_sections = 5
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 7: 3D uint8 array, split into 1 section
    ary = np.random.randint(0, 255, size=(10, 10, 1)).astype(np.uint8)
    indices_or_sections = 1
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 8: 4D float32 array with zeros, split into 2 sections
    ary = np.zeros((3, 3, 4, 5), dtype=np.float32)
    indices_or_sections = 2
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 9: 3D float16 array, split into 6 sections
    ary = np.random.randn(2, 3, 12).astype(np.float16)
    indices_or_sections = 6
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 10: 6D int16 array, split into 2 sections
    ary = np.random.randint(-100, 100, size=(1, 2, 2, 1, 2, 3)).astype(np.int16)
    indices_or_sections = 2
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    # Input 11: 3D float32 array with ones, split into 8 sections
    ary = np.ones((5, 1, 16), dtype=np.float32)
    indices_or_sections = 8
    list_of_inputs.append(copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections}))

    return list_of_inputs

generated_inputs["jax.numpy.dsplit_1"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dsplit_1'.")


check_valid('jax.numpy.dsplit', generated_inputs['jax.numpy.dsplit_1'], lib="jax", suffix=1)
