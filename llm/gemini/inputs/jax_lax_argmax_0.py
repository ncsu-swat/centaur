
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_argmax_inputs():
    list_of_inputs = []

    # Input 1: 1D Float32 tensor, axis 0
    operand = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    axis = 0
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 2: 2D Float32 tensor, axis 1
    operand = np.random.randn(4, 5).astype(np.float32)
    axis = 1
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 3: 2D Int32 tensor, axis 0
    operand = np.random.randint(-10, 10, size=(3, 3)).astype(np.int32)
    axis = 0
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 4: 3D Float64 tensor, axis 2
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 2
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 5: 4D Float32 tensor, axis 2
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = 2
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 6: 3D Int16 tensor, axis 1
    operand = np.random.randint(-100, 100, size=(2, 4, 2)).astype(np.int16)
    axis = 1
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 7: 1D Float16 tensor, axis 0
    operand = np.array([-1.5, -3.0, -0.5, -2.2], dtype=np.float16)
    axis = 0
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 8: 3D Float32 tensor, axis 0
    operand = np.random.randn(3, 5, 2).astype(np.float32)
    axis = 0
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 9: 2D Boolean tensor, axis 0
    operand = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    axis = 0
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    # Input 10: 2D Float32 tensor, axis 1
    operand = np.random.randn(10, 20).astype(np.float32)
    axis = 1
    index_dtype = np.dtype('int32')
    list_of_inputs.append({"operand": operand, "axis": axis, "index_dtype": index_dtype})

    return list_of_inputs

generated_inputs["jax.lax.argmax"] = jax_lax_argmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.argmax'.")


check_valid('jax.lax.argmax', generated_inputs['jax.lax.argmax'], lib="jax", suffix=0)
