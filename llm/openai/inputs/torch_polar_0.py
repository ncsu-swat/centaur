
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def polar_inputs():
    list_of_inputs = []
    
    abs1 = torch.tensor([1.0, 2.0], dtype=torch.float64).numpy()
    angle1 = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64).numpy()
    out1 = torch.tensor([], dtype=torch.complex128).numpy()
    input_dict1 = {"abs": abs1, "angle": angle1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    abs2 = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32).numpy()
    angle2 = torch.tensor([0.0, np.pi / 4, np.pi / 2], dtype=torch.float32).numpy()
    out2 = torch.tensor([], dtype=torch.complex64).numpy()
    input_dict2 = {"abs": abs2, "angle": angle2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    abs3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64).numpy()
    angle3 = torch.tensor([[0.0, np.pi / 2], [np.pi / 4, np.pi]], dtype=torch.float64).numpy()
    out3 = torch.tensor([], dtype=torch.complex128).numpy()
    input_dict3 = {"abs": abs3, "angle": angle3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    abs4 = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32).numpy()
    angle4 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).numpy()
    out4 = torch.tensor([], dtype=torch.complex64).numpy()
    input_dict4 = {"abs": abs4, "angle": angle4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    abs5 = torch.tensor([1.0], dtype=torch.float64).numpy()
    angle5 = torch.tensor([np.pi], dtype=torch.float64).numpy()
    out5 = torch.tensor([], dtype=torch.complex128).numpy()
    input_dict5 = {"abs": abs5, "angle": angle5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    abs6 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    angle6 = torch.tensor([np.pi / 3, np.pi / 6, np.pi / 4], dtype=torch.float32).numpy()
    out6 = torch.tensor([], dtype=torch.complex64).numpy()
    input_dict6 = {"abs": abs6, "angle": angle6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    abs7 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64).numpy()
    angle7 = torch.tensor([[0.0, np.pi/2, np.pi], [np.pi/4, 3*np.pi/4, np.pi/2]], dtype=torch.float64).numpy()
    out7 = torch.tensor([], dtype=torch.complex128).numpy()
    input_dict7 = {"abs": abs7, "angle": angle7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    abs8 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).numpy()
    angle8 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).numpy()
    out8 = torch.tensor([], dtype=torch.complex64).numpy()
    input_dict8 = {"abs": abs8, "angle": angle8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    abs9 = torch.tensor([1.5], dtype=torch.float64).numpy()
    angle9 = torch.tensor([2*np.pi], dtype=torch.float64).numpy()
    out9 = torch.tensor([], dtype=torch.complex128).numpy()
    input_dict9 = {"abs": abs9, "angle": angle9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    abs10 = torch.tensor([0.1, 0.2], dtype=torch.float32).numpy()
    angle10 = torch.tensor([np.pi / 2, 0.0], dtype=torch.float32).numpy()
    out10 = torch.tensor([], dtype=torch.complex64).numpy()
    input_dict10 = {"abs": abs10, "angle": angle10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.polar"] = polar_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.polar' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.polar'.")


check_valid('torch.polar', generated_inputs['torch.polar'], lib="torch", suffix=0)
