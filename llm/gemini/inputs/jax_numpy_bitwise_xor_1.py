
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # 1. 1D int32
    list_of_inputs.append({"x": np.array([1, 2], dtype=np.int32), "y": np.array([3, 4], dtype=np.int32)})
    # 2. 2D int32
    list_of_inputs.append({"x": np.array([[1]], dtype=np.int32), "y": np.array([[2]], dtype=np.int32)})
    # 3. Boolean
    list_of_inputs.append({"x": np.array([True], dtype=bool), "y": np.array([False], dtype=bool)})
    # 4. int64 with negatives
    list_of_inputs.append({"x": np.array([-1], dtype=np.int64), "y": np.array([2], dtype=np.int64)})
    # 5. 3D int16
    list_of_inputs.append({"x": np.array([[[1]]], dtype=np.int16), "y": np.array([[[2]]], dtype=np.int16)})
    # 6. Scalar-like 0D
    list_of_inputs.append({"x": np.array(5, dtype=np.int32), "y": np.array(3, dtype=np.int32)})
    # 7. uint8 with broadcasting
    list_of_inputs.append({"x": np.array([[1, 2]], dtype=np.uint8), "y": np.array([1], dtype=np.uint8)})
    # 8. Boolean 2D
    list_of_inputs.append({"x": np.array([[True]], dtype=bool), "y": np.array([[False]], dtype=bool)})
    # 9. int8
    list_of_inputs.append({"x": np.array([-5], dtype=np.int8), "y": np.array([5], dtype=np.int8)})
    # 10. 4D int32
    list_of_inputs.append({"x": np.zeros((1, 1, 1, 1), dtype=np.int32), "y": np.ones((1, 1, 1, 1), dtype=np.int32)})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.bitwise_xor_1"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_1'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_1'], lib="jax", suffix=1)
