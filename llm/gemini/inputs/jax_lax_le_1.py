
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_le_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 comparison, 1D
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Int32 comparison with negative values, 2D
    x = np.array([[-5, 10], [0, -3]], dtype=np.int32)
    y = np.array([[-4, 10], [1, -5]], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 comparison, 3D
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting, (3, 1) and (1, 4) - both 2D
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int8 comparison, 1D
    x = np.array([-128, 0, 127], dtype=np.int8)
    y = np.array([-127, 0, 126], dtype=np.int8)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Unsigned integers uint8, 1D
    x = np.array([0, 255, 128], dtype=np.uint8)
    y = np.array([1, 255, 127], dtype=np.uint8)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float16 comparison, 4D
    x = np.random.randn(2, 2, 2, 2).astype(np.float16)
    y = np.random.randn(2, 2, 2, 2).astype(np.float16)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D Boolean array comparison
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large arrays, 1D int64
    x = np.arange(100, dtype=np.int64)
    y = np.arange(100, dtype=np.int64) + 1
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting with matching rank, (1, 1) and (3, 3) - both 2D
    x = np.array([[5.5]], dtype=np.float32)
    y = np.random.randn(3, 3).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.le_1"] = jax_lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_1'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_1'], lib="jax", suffix=1)
