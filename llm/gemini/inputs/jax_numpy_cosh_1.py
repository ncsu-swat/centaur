
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cosh_inputs():
    list_of_inputs = []

    # 1. 0-D scalar float32 array
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 2. 1-D float64 array with positive and negative values
    x = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 3. 2-D integer array
    x = np.array([[1, -2, 3], [0, 4, -5]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 4. 2-D float16 array
    x = np.random.uniform(-2.0, 2.0, size=(4, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 5. 3-D float32 array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 6. 1-D complex64 array
    x = np.array([1.0 + 2.0j, -0.5 - 1.5j, 0.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 7. 2-D complex128 array
    x = np.array(
        [[1.0 - 1.0j, 2.0 + 3.0j], [-2.0 + 0.5j, 0.0 + 0.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 8. 4-D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 9. 1-D boolean array
    x = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 10. 1-D int64 array with zero and small values
    x = np.array([0, -10, 10, -5, 5], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # 11. Empty array
    x = np.empty((0, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.cosh_1"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cosh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cosh_1'.")


check_valid('jax.numpy.cosh', generated_inputs['jax.numpy.cosh_1'], lib="jax", suffix=1)
