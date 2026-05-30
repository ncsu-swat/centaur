
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dsplit_inputs():
    list_of_inputs = []

    # Case 1: 3D float32, indices (2, 4)
    ary = np.random.randn(2, 2, 6).astype(np.float32)
    indices_or_sections = (2, 4)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 2: 3D int32, indices (4,)
    ary = np.random.randint(0, 10, size=(3, 4, 8)).astype(np.int32)
    indices_or_sections = (4,)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 3: 4D float64, indices (1, 3)
    ary = np.random.randn(2, 2, 4, 3).astype(np.float64)
    indices_or_sections = (1, 3)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 4: 3D uint8, indices (2, 5, 8)
    ary = np.random.randint(0, 255, size=(1, 1, 10)).astype(np.uint8)
    indices_or_sections = (2, 5, 8)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 5: 3D float16, indices (1, 2, 3, 4)
    ary = np.random.randn(2, 3, 5).astype(np.float16)
    indices_or_sections = (1, 2, 3, 4)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 6: 5D int64, indices (3,)
    ary = np.random.randint(-100, 100, size=(2, 2, 6, 2, 2)).astype(np.int64)
    indices_or_sections = (3,)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 7: 3D float32, indices (3, 6, 9)
    ary = np.random.randn(4, 4, 12).astype(np.float32)
    indices_or_sections = (3, 6, 9)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 8: 3D complex64, indices (1, 2)
    ary = (np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)).astype(np.complex64)
    indices_or_sections = (1, 2)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 9: 3D bool, indices (1,)
    ary = np.random.choice([True, False], size=(1, 1, 2)).astype(np.bool_)
    indices_or_sections = (1,)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    # Case 10: 4D float32, indices (5, 10)
    ary = np.random.randn(3, 3, 15, 2).astype(np.float32)
    indices_or_sections = (5, 10)
    list_of_inputs.append({"ary": ary, "indices_or_sections": indices_or_sections})

    return list_of_inputs

generated_inputs["jax.numpy.dsplit_3"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dsplit_3'.")


check_valid('jax.numpy.dsplit', generated_inputs['jax.numpy.dsplit_3'], lib="jax", suffix=3)
