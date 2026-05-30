
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unwrap_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array, standard 2*pi period
    p = np.array([0.0, 0.5, 1.0, 5.0, 5.5, 6.0], dtype=np.float32)
    discont = np.array(np.pi, dtype=np.float32)
    axis = 0
    period = np.array(2 * np.pi, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array in degrees (period 360)
    p = np.array([179.0, -179.0, 180.0, -180.0], dtype=np.float32)
    discont = np.array(180.0, dtype=np.float32)
    axis = -1
    period = np.array(360.0, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, unwrapping along axis 0
    p = np.array([[0.0, 1.0], [5.0, 6.0], [0.1, 1.1]], dtype=np.float32)
    discont = np.array(np.pi, dtype=np.float32)
    axis = 0
    period = np.array(2 * np.pi, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, unwrapping along axis 1
    p = np.array([[0.0, 5.0, 0.1], [1.0, 6.0, 1.1]], dtype=np.float32)
    discont = np.array(np.pi, dtype=np.float32)
    axis = 1
    period = np.array(2 * np.pi, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 precision 1D array
    p = np.array([-3.0, -1.0, 2.0, 4.0, -4.0], dtype=np.float64)
    discont = np.array(3.0, dtype=np.float64)
    axis = 0
    period = np.array(6.0, dtype=np.float64)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, unwrapping along axis 2
    p = np.random.uniform(-np.pi, np.pi, size=(2, 3, 4)).astype(np.float32)
    discont = np.array(np.pi, dtype=np.float32)
    axis = 2
    period = np.array(2 * np.pi, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small period and small discontinuity
    p = np.array([0.01, 0.02, 0.08, 0.01, 0.02], dtype=np.float32)
    discont = np.array(0.05, dtype=np.float32)
    axis = 0
    period = np.array(0.1, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with negative axis specification
    p = np.array([0.0, 0.9, 1.8, 0.1, 0.8], dtype=np.float32)
    discont = np.array(0.5, dtype=np.float32)
    axis = -1
    period = np.array(1.0, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 2D array with large values
    p = np.array([[100.0, 105.0, 110.0], [200.0, 201.0, 195.0]], dtype=np.float64)
    discont = np.array(5.0, dtype=np.float64)
    axis = 1
    period = np.array(10.0, dtype=np.float64)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, unwrapping along axis 1
    p = np.random.uniform(-10.0, 10.0, size=(2, 5, 2, 2)).astype(np.float32)
    discont = np.array(10.0, dtype=np.float32)
    axis = 1
    period = np.array(20.0, dtype=np.float32)
    input_dict = {"p": p, "discont": discont, "axis": axis, "period": period}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unwrap_2"] = unwrap_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unwrap_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unwrap_2'.")


check_valid('jax.numpy.unwrap', generated_inputs['jax.numpy.unwrap_2'], lib="jax", suffix=2)
