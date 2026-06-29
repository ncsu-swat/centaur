
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log_ndtr_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, standard range
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "series_order": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32, very negative values (below float32 lower_segment of -10)
    x = np.array([-15.0, -12.0, -10.5], dtype=np.float32)
    input_dict = {"x": x, "series_order": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32, positive values (above float32 upper_segment of 5)
    x = np.array([6.0, 10.0, 20.0], dtype=np.float32)
    input_dict = {"x": x, "series_order": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64, mixed range
    x = np.random.uniform(-5.0, 5.0, size=(3, 3)).astype(np.float64)
    input_dict = {"x": x, "series_order": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32, series_order 0
    x = np.random.uniform(-1.0, 1.0, size=(2, 4)).astype(np.float32)
    input_dict = {"x": x, "series_order": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32, mixed range
    x = np.random.normal(size=(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "series_order": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float64, max supported series_order
    x = np.random.normal(size=(2, 1, 3, 2)).astype(np.float64)
    input_dict = {"x": x, "series_order": 30}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D float32 (scalar array)
    x = np.array(-3.5, dtype=np.float32)
    input_dict = {"x": x, "series_order": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64, wide range with float64 segment limits (-20 and 8)
    x = np.array([-30.0, -21.0, -5.0, 0.0, 5.0, 9.0, 30.0], dtype=np.float64)
    input_dict = {"x": x, "series_order": 15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32, extreme negative values
    x = np.random.uniform(-25.0, -10.0, size=(4, 2)).astype(np.float32)
    input_dict = {"x": x, "series_order": 8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.log_ndtr"] = log_ndtr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.log_ndtr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.log_ndtr'.")


check_valid('jax.scipy.special.log_ndtr', generated_inputs['jax.scipy.special.log_ndtr'], lib="jax", suffix=0)
