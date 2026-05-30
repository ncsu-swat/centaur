
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dsplit_inputs():
    list_of_inputs = []

    # Input 1: 3D array of floats, size along axis 2 is 4, split into 2
    ary = np.random.randn(2, 2, 4).astype(np.float32)
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array of ints, size along axis 2 is 6, split into 3
    ary = np.random.randint(-10, 10, size=(3, 1, 6)).astype(np.int32)
    indices_or_sections = 3
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D array of floats, size along axis 2 is 8, split into 2
    ary = np.random.randn(1, 2, 8, 3).astype(np.float32)
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array of bools, size along axis 2 is 2, split into 2
    ary = np.random.choice([True, False], size=(2, 2, 2))
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of float64, size along axis 2 is 12, split into 4
    ary = np.random.randn(4, 4, 12).astype(np.float64)
    indices_or_sections = 4
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of complex numbers, size along axis 2 is 4, split into 2
    ary = (np.random.randn(1, 1, 4) + 1j * np.random.randn(1, 1, 4)).astype(np.complex64)
    indices_or_sections = 2
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D array of ints, size along axis 2 is 10, split into 5
    ary = np.random.randint(0, 100, size=(2, 2, 10, 2, 2)).astype(np.int64)
    indices_or_sections = 5
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of floats, size along axis 2 is 1, split into 1
    ary = np.random.randn(3, 3, 1).astype(np.float32)
    indices_or_sections = 1
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with negative values, size along axis 2 is 9, split into 3
    ary = np.random.uniform(-5.0, 5.0, size=(2, 3, 9)).astype(np.float32)
    indices_or_sections = 3
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array of floats, size along axis 2 is 6, split into 6
    ary = np.random.randn(2, 2, 6, 2).astype(np.float32)
    indices_or_sections = 6
    input_dict = {"ary": ary, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
