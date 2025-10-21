
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def isclose_inputs():
    list_of_inputs = []
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input2 = np.array([1.0 + 1e-10, 3.0, 4.0], dtype=np.float32)
    rtol1 = 1e-05
    atol1 = 1e-08
    equal_nan1 = False
    input_dict1 = {"input": input1, "other": input2, "rtol": rtol1, "atol": atol1, "equal_nan": equal_nan1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input3 = np.array([float('inf'), 4.0], dtype=np.float64)
    input4 = np.array([float('inf'), 6.0], dtype=np.float64)
    rtol2 = .5
    atol2 = 1e-08
    equal_nan2 = False
    input_dict2 = {"input": input3, "other": input4, "rtol": rtol2, "atol": atol2, "equal_nan": equal_nan2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input5 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input6 = np.array([-1.0 - 1e-10, -3.0, -4.0], dtype=np.float32)
    rtol3 = 1e-05
    atol3 = 1e-08
    equal_nan3 = False
    input_dict3 = {"input": input5, "other": input6, "rtol": rtol3, "atol": atol3, "equal_nan": equal_nan3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input7 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input8 = np.array([[1.0 + 1e-10, 2.0], [3.0, 4.0]], dtype=np.float32)
    rtol4 = 1e-05
    atol4 = 1e-08
    equal_nan4 = False
    input_dict4 = {"input": input7, "other": input8, "rtol": rtol4, "atol": atol4, "equal_nan": equal_nan4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input9 = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    input10 = np.array([np.nan, 3.0, 4.0], dtype=np.float32)
    rtol5 = 1e-05
    atol5 = 1e-08
    equal_nan5 = True
    input_dict5 = {"input": input9, "other": input10, "rtol": rtol5, "atol": atol5, "equal_nan": equal_nan5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input11 = np.array([np.nan, 2.0, 3.0], dtype=np.float32)
    input12 = np.array([np.nan, 3.0, 4.0], dtype=np.float32)
    rtol6 = 1e-05
    atol6 = 1e-08
    equal_nan6 = False
    input_dict6 = {"input": input11, "other": input12, "rtol": rtol6, "atol": atol6, "equal_nan": equal_nan6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input13 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input14 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rtol7 = 0.1
    atol7 = 0.1
    equal_nan7 = False
    input_dict7 = {"input": input13, "other": input14, "rtol": rtol7, "atol": atol7, "equal_nan": equal_nan7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input15 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input16 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    rtol8 = 1e-05
    atol8 = 1e-08
    equal_nan8 = False
    input_dict8 = {"input": input15, "other": input16, "rtol": rtol8, "atol": atol8, "equal_nan": equal_nan8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input17 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input18 = np.array([1.0, 2.0, 3.1], dtype=np.float32)
    rtol9 = 1e-05
    atol9 = 1e-07
    equal_nan9 = False
    input_dict9 = {"input": input17, "other": input18, "rtol": rtol9, "atol": atol9, "equal_nan": equal_nan9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input19 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input20 = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    rtol10 = 0.1
    atol10 = 0.0
    equal_nan10 = False
    input_dict10 = {"input": input19, "other": input20, "rtol": rtol10, "atol": atol10, "equal_nan": equal_nan10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.isclose"] = isclose_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isclose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isclose'.")


check_valid('torch.isclose', generated_inputs['torch.isclose'], lib="torch", suffix=0)
