
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rev_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reverse along dimension 0
    operand = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dimensions = [0]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, reverse along dimension 0
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    dimensions = [0]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, reverse along dimension 1
    operand = np.random.randn(4, 5).astype(np.float32)
    dimensions = [1]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, reverse along dimensions 0 and 1
    operand = np.random.randn(3, 3).astype(np.float64)
    dimensions = [0, 1]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int16 array, reverse along dimension 2
    operand = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int16)
    dimensions = [2]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, reverse along dimensions 0 and 2
    operand = np.random.randn(2, 2, 2).astype(np.float32)
    dimensions = [0, 2]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D bool array, reverse along dimension 0
    operand = np.array([True, False, True, True, False], dtype=np.bool_)
    dimensions = [0]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, reverse along dimension 1 and 3
    operand = np.random.randn(2, 3, 2, 4).astype(np.float32)
    dimensions = [1, 3]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex64 array, reverse along dimension 0
    operand = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    dimensions = [0]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int32 array, reverse along dimensions 1, 3, 4
    operand = np.random.randint(-5, 5, size=(2, 2, 2, 2, 2)).astype(np.int32)
    dimensions = [1, 3, 4]
    input_dict = {"operand": operand, "dimensions": dimensions}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.rev_1"] = rev_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rev_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rev_1'.")


check_valid('jax.lax.rev', generated_inputs['jax.lax.rev_1'], lib="jax", suffix=1)
