
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def geomspace_inputs():
    list_of_inputs = []

    # Input 1: 0D arrays, positive floats
    start = np.array(1.0, dtype=np.float32)
    stop = np.array(1000.0, dtype=np.float32)
    num = 10
    endpoint = True
    dtype = np.dtype(np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 2: 0D arrays, negative floats
    start = np.array(-1.0, dtype=np.float64)
    stop = np.array(-100.0, dtype=np.float64)
    num = 5
    endpoint = False
    dtype = np.dtype(np.float64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 3: 1D arrays, axis=0
    start = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    stop = np.array([16.0, 32.0, 48.0], dtype=np.float32)
    num = 8
    endpoint = True
    dtype = np.dtype(np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 4: 1D arrays, axis=1, endpoint=False
    start = np.array([2.0, 5.0], dtype=np.float64)
    stop = np.array([128.0, 625.0], dtype=np.float64)
    num = 4
    endpoint = False
    dtype = np.dtype(np.float64)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 5: 1D arrays, axis=-1
    start = np.array([10.0, 100.0, 1000.0], dtype=np.float32)
    stop = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    num = 4
    endpoint = True
    dtype = np.dtype(np.float32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 6: 2D arrays, axis=0
    start = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    stop = np.array([[8.0, 16.0], [24.0, 32.0]], dtype=np.float32)
    num = 4
    endpoint = True
    dtype = np.dtype(np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 7: 2D arrays, axis=1, endpoint=False
    start = np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float64)
    stop = np.array([[1000.0, 100.0], [10.0, 1.0]], dtype=np.float64)
    num = 6
    endpoint = False
    dtype = np.dtype(np.float64)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 8: 2D arrays, axis=2
    start = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    stop = np.array([[16.0, 81.0, 256.0]], dtype=np.float32)
    num = 5
    endpoint = True
    dtype = np.dtype(np.float32)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 9: 3D arrays, axis=0
    start = np.full((2, 2, 2), 1.0, dtype=np.float32)
    stop = np.full((2, 2, 2), 64.0, dtype=np.float32)
    num = 7
    endpoint = True
    dtype = np.dtype(np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    # Input 10: 1D complex numbers, axis=0
    start = np.array([1.0 + 1.0j, 2.0 + 0.0j], dtype=np.complex64)
    stop = np.array([4.0 + 4.0j, 8.0 + 0.0j], dtype=np.complex64)
    num = 3
    endpoint = True
    dtype = np.dtype(np.complex64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "start": start,
                "stop": stop,
                "num": num,
                "endpoint": endpoint,
                "dtype": dtype,
                "axis": axis,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.geomspace_1"] = geomspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.geomspace_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.geomspace_1'.")


check_valid('jax.numpy.geomspace', generated_inputs['jax.numpy.geomspace_1'], lib="jax", suffix=1)
