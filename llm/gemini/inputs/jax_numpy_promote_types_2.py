
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def promote_types_inputs():
    list_of_inputs = []

    # Input 1: int32 and float32
    input_dict = {"a": "int32", "b": "float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: bool and int8
    input_dict = {"a": "bool", "b": "int8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: uint8 and int16
    input_dict = {"a": "uint8", "b": "int16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float16 and float32
    input_dict = {"a": "float16", "b": "float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32 and complex64
    input_dict = {"a": "float32", "b": "complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64 and uint64
    input_dict = {"a": "int64", "b": "uint64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 and float32
    input_dict = {"a": "bfloat16", "b": "float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Character codes: 'f4' and 'f8'
    input_dict = {"a": "f4", "b": "f8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 'i4' and 'u4'
    input_dict = {"a": "i4", "b": "u4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 and complex128
    input_dict = {"a": "float64", "b": "complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint32 and float32
    input_dict = {"a": "uint32", "b": "float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: bool and complex64
    input_dict = {"a": "bool", "b": "complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.promote_types_2"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.promote_types_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.promote_types_2'.")


check_valid('jax.numpy.promote_types', generated_inputs['jax.numpy.promote_types_2'], lib="jax", suffix=2)
