
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def convert_element_type_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 to int32
    operand = np.array([-2.5, -1.0, 0.0, 1.7, 3.2], dtype=np.float32)
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 2: 2D int32 to float32
    operand = np.array([[-10, 20], [0, -5]], dtype=np.int32)
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 3: 0D scalar float32 to bool
    operand = np.array(0.0, dtype=np.float32)
    new_dtype = np.dtype(np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 4: 3D bool to int32
    operand = np.array(
        [[[True, False], [False, True]], [[True, True], [False, False]]],
        dtype=np.bool_,
    )
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 5: 2D int32 to bool
    operand = np.array([[0, 1], [-5, 0]], dtype=np.int32)
    new_dtype = np.dtype(np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 6: 1D bool to float32
    operand = np.array([True, False, True, True], dtype=np.bool_)
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 7: 4D float32 to int32
    operand = np.ones((2, 1, 3, 2), dtype=np.float32) * -15.5
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 8: 2D float32 to float32 (identity cast)
    operand = np.array([[1.0, -2.0], [3.5, 0.0]], dtype=np.float32)
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 9: 3D int32 to int32 (identity cast)
    operand = np.zeros((2, 3, 4), dtype=np.int32)
    new_dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

    # Input 10: 1D int32 with negative and large values to float32
    operand = np.array([-100000, 0, 100000], dtype=np.int32)
    new_dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "new_dtype": new_dtype})
    )

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
