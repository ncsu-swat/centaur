
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sort_inputs():
    list_of_inputs = []

    # 1. 1D float32 array, default-like settings
    a = np.array([3.0, -1.0, 2.5, 0.0, -4.2], dtype=np.float32)
    input_dict = {"a": a, "axis": -1, "stable": True, "descending": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. 1D int32 array, descending
    a = np.array([10, -5, 20, 0, 7, -12], dtype=np.int32)
    input_dict = {"a": a, "axis": 0, "stable": True, "descending": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. 2D float64 array, sort along axis 0
    a = np.array(
        [[1.5, -2.0, 3.3], [-0.5, 4.2, 0.0], [2.1, 1.1, -3.4]],
        dtype=np.float64,
    )
    input_dict = {"a": a, "axis": 0, "stable": False, "descending": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. 2D float32 array, sort along axis 1, descending
    a = np.array([[5.0, 2.0, 9.0], [-1.0, 3.0, 0.0]], dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "stable": True, "descending": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. 2D int64 array, negative axis index
    a = np.array([[12, -3, 8], [4, 15, -9]], dtype=np.int64)
    input_dict = {"a": a, "axis": -2, "stable": False, "descending": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. 3D float32 array, sort along axis 2
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "axis": 2, "stable": True, "descending": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. 3D int16 array, sort along axis 1, descending
    a = np.random.randint(-100, 100, size=(3, 4, 2)).astype(np.int16)
    input_dict = {"a": a, "axis": 1, "stable": False, "descending": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. 4D float32 array, sort along axis 0
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {"a": a, "axis": 0, "stable": True, "descending": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Boolean array, sort along axis -1
    a = np.array([[True, False, True], [False, False, True]], dtype=bool)
    input_dict = {"a": a, "axis": -1, "stable": True, "descending": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. uint8 array with duplicates, stable sort
    a = np.array([255, 0, 128, 0, 64, 255, 128], dtype=np.uint8)
    input_dict = {"a": a, "axis": 0, "stable": True, "descending": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11. 3D float64 array, sort along axis -1, descending
    a = np.random.randn(4, 1, 5).astype(np.float64)
    input_dict = {"a": a, "axis": -1, "stable": False, "descending": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.sort_1"] = sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sort_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sort_1'.")


check_valid('jax.numpy.sort', generated_inputs['jax.numpy.sort_1'], lib="jax", suffix=1)
