
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def linspace_inputs():
    list_of_inputs = []

    # Input 1: Basic scalar inputs (as 0D arrays)
    list_of_inputs.append({
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(10.0, dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype('float32'),
        "axis": 0
    })

    # Input 2: 1D arrays with endpoint=False and retstep=True
    list_of_inputs.append({
        "start": np.array([0.0, 5.0], dtype=np.float32),
        "stop": np.array([10.0, 15.0], dtype=np.float32),
        "num": 10,
        "endpoint": False,
        "retstep": True,
        "dtype": np.dtype('float32'),
        "axis": 0
    })

    # Input 3: Float64 arrays with negative values
    list_of_inputs.append({
        "start": np.array([-5.0, -10.0], dtype=np.float64),
        "stop": np.array([5.0, 10.0], dtype=np.float64),
        "num": 20,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype('float64'),
        "axis": 0
    })

    # Input 4: 2D arrays, using negative axis (-1)
    list_of_inputs.append({
        "start": np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        "stop": np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32),
        "num": 5,
        "endpoint": True,
        "retstep": True,
        "dtype": np.dtype('float32'),
        "axis": -1
    })

    # Input 5: Integer source arrays, float output dtype
    list_of_inputs.append({
        "start": np.array([0, 10], dtype=np.int32),
        "stop": np.array([100, 110], dtype=np.int32),
        "num": 8,
        "endpoint": False,
        "retstep": False,
        "dtype": np.dtype('float32'),
        "axis": 0
    })

    # Input 6: 0D float64 arrays with retstep
    list_of_inputs.append({
        "start": np.array(-1.5, dtype=np.float64),
        "stop": np.array(1.5, dtype=np.float64),
        "num": 15,
        "endpoint": True,
        "retstep": True,
        "dtype": np.dtype('float64'),
        "axis": 0
    })

    # Input 7: Multi-dimensional arrays with axis=1
    list_of_inputs.append({
        "start": np.ones((2, 3), dtype=np.float32),
        "stop": np.ones((2, 3), dtype=np.float32) * 10,
        "num": 4,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype('float32'),
        "axis": 1
    })

    # Input 8: 3D arrays with axis=2, endpoint=False
    list_of_inputs.append({
        "start": np.zeros((2, 2, 2), dtype=np.float32),
        "stop": np.ones((2, 2, 2), dtype=np.float32) * 5,
        "num": 6,
        "endpoint": False,
        "retstep": True,
        "dtype": np.dtype('float32'),
        "axis": 2
    })

    # Input 9: Float16 precision arrays
    list_of_inputs.append({
        "start": np.array([1.0], dtype=np.float16),
        "stop": np.array([2.0], dtype=np.float16),
        "num": 3,
        "endpoint": True,
        "retstep": False,
        "dtype": np.dtype('float16'),
        "axis": 0
    })

    # Input 10: Single element interval generation (num=1)
    list_of_inputs.append({
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(1.0, dtype=np.float32),
        "num": 1,
        "endpoint": True,
        "retstep": True,
        "dtype": np.dtype('float32'),
        "axis": 0
    })

    return list_of_inputs

generated_inputs["jax.numpy.linspace_2"] = linspace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linspace_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linspace_2'.")


check_valid('jax.numpy.linspace', generated_inputs['jax.numpy.linspace_2'], lib="jax", suffix=2)
