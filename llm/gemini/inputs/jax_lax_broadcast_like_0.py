
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_like_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D int32
    arr = np.array([1, 2, 3], dtype=np.int32)
    like_arr = np.zeros((2, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 2: 0D scalar array to 3D float32 with negative values
    arr = np.array(-5.5, dtype=np.float32)
    like_arr = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 3: 2D with singleton dimension to 2D float64
    arr = np.array([[-1.2, 0.5, 3.1, -4.0]], dtype=np.float64)
    like_arr = np.ones((5, 4), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 4: 2D with singleton dimension to 3D int64
    arr = np.array([[10], [-20], [30]], dtype=np.int64)
    like_arr = np.empty((2, 3, 5), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 5: 1D boolean array to 2D boolean
    arr = np.array([True, False, True], dtype=np.bool_)
    like_arr = np.zeros((4, 3), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 6: 3D singleton array to 3D complex64
    arr = np.array([[[1.0 + 2.0j]]], dtype=np.complex64)
    like_arr = np.zeros((4, 3, 2), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 7: Same shape broadcasting float32
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    like_arr = np.zeros((3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 8: 0D uint8 to 1D uint8
    arr = np.array(255, dtype=np.uint8)
    like_arr = np.zeros((10,), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 9: 1D float32 to 4D float32
    arr = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    like_arr = np.zeros((2, 2, 2, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 10: 2D int16 with singleton dimension to 2D int16
    arr = np.array([[-3], [0], [7], [-12]], dtype=np.int16)
    like_arr = np.zeros((4, 6), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    # Input 11: 3D array broadcasting across middle dimension
    arr = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
    like_arr = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"arr": arr, "like_arr": like_arr}))

    return list_of_inputs


generated_inputs["jax.lax.broadcast_like"] = broadcast_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_like'.")


check_valid('jax.lax.broadcast_like', generated_inputs['jax.lax.broadcast_like'], lib="jax", suffix=0)
