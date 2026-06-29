
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_broadcast_inputs():
    list_of_inputs = []

    # Input 1: Float32, 2D operand, 2D sizes
    operand = np.random.randn(4, 5).astype(np.float32)
    sizes = (2, 3)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, 1D operand, 1D sizes
    operand = np.random.randn(5).astype(np.float64)
    sizes = (10,)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, 2D square operand, 2D sizes
    operand = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    sizes = (1, 2)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int64, 1D operand, 3D sizes
    operand = np.random.randint(0, 100, size=(2,)).astype(np.int64)
    sizes = (2, 2, 2)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float16, 3D operand, 1D sizes
    operand = np.random.randn(2, 2, 2).astype(np.float16)
    sizes = (5,)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Bool, 1D operand, 2D sizes
    operand = np.random.choice([True, False], size=(4,)).astype(np.bool_)
    sizes = (2, 4)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64, 1D operand, 2D sizes
    operand = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    sizes = (3, 2)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float32, 0D operand (scalar), 2D sizes
    operand = np.array(3.14, dtype=np.float32)
    sizes = (4, 4)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uint8, 3D operand with singleton dimension, 2D sizes
    operand = np.random.randint(0, 255, size=(3, 1, 2)).astype(np.uint8)
    sizes = (5, 6)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16, 2D operand, 1D sizes
    operand = np.random.randint(-50, 50, size=(5, 5)).astype(np.int16)
    sizes = (12,)
    input_dict = {"operand": operand, "sizes": sizes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.broadcast_1"] = jax_lax_broadcast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_1'.")


check_valid('jax.lax.broadcast', generated_inputs['jax.lax.broadcast_1'], lib="jax", suffix=1)
