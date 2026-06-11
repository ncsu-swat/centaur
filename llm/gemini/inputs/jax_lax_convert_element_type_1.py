
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def convert_element_type_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative values to int32
    operand = np.array([-2.5, -1.0, 0.0, 1.5, 2.8], dtype=np.float32)
    new_dtype = np.dtype(np.int32)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array to float32
    operand = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    new_dtype = np.dtype(np.float32)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array to float16
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    new_dtype = np.dtype(np.float16)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D (scalar) float32 array to int64
    operand = np.array(-3.14, dtype=np.float32)
    new_dtype = np.dtype(np.int64)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D uint8 array to float32
    operand = np.random.randint(0, 256, size=(1, 3, 5, 5)).astype(np.uint8)
    new_dtype = np.dtype(np.float32)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float32 array to complex64
    operand = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    new_dtype = np.dtype(np.complex64)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int16 array to bool
    operand = np.array([[0, 1], [-1, 0]], dtype=np.int16)
    new_dtype = np.dtype(np.bool_)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D bool array to int8
    operand = np.random.choice([True, False], size=(1, 2, 2, 2, 2))
    new_dtype = np.dtype(np.int8)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array to uint32
    operand = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    new_dtype = np.dtype(np.uint32)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int64 array to int32
    operand = np.array([[[1000, -2000], [3000, -4000]]], dtype=np.int64)
    new_dtype = np.dtype(np.int32)
    input_dict = {"operand": operand, "new_dtype": new_dtype}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.convert_element_type_1"] = convert_element_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.convert_element_type_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.convert_element_type_1'.")


check_valid('jax.lax.convert_element_type', generated_inputs['jax.lax.convert_element_type_1'], lib="jax", suffix=1)
