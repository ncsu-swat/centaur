
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def poisson_nll_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0])
    target1 = np.array([0, 1, 2])
    input_dict1 = {
        "input": input1,
        "target": target1,
        "log_input": False,
        "full": False,
        "eps": 1e-8,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    target2 = np.array([[0, 1], [2, 3]])
    input_dict2 = {
        "input": input2,
        "target": target2,
        "log_input": True,
        "full": True,
        "eps": 1e-5,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.1, 0.2, 0.3, 0.4])
    target3 = np.array([5, 2, 1, 0])
    input_dict3 = {
        "input": input3,
        "target": target3,
        "log_input": False,
        "full": False,
        "eps": 1e-7,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    target4 = np.array([[1, 0], [0, 2]])
    input_dict4 = {
        "input": input4,
        "target": target4,
        "log_input": True,
        "full": False,
        "eps": 1e-6,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(2, 2, 2)
    target5 = np.random.randint(0, 5, size=(2, 2, 2))
    input_dict5 = {
        "input": input5,
        "target": target5,
        "log_input": False,
        "full": True,
        "eps": 1e-9,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0])
    target6 = np.array([0])
    input_dict6 = {
        "input": input6,
        "target": target6,
        "log_input": False,
        "full": False,
        "eps": 1e-8,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([10.0, 20.0, 30.0])
    target7 = np.array([1, 2, 3])
    input_dict7 = {
        "input": input7,
        "target": target7,
        "log_input": True,
        "full": False,
        "eps": 1e-5,
        "reduction": "none"
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(3, 4, 5)
    target8 = np.random.randint(0, 10, size=(3, 4, 5))
    input_dict8 = {
        "input": input8,
        "target": target8,
        "log_input": False,
        "full": True,
        "eps": 1e-7,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0.5, 1.5, 2.5])
    target9 = np.array([0, 1, 2])
    input_dict9 = {
        "input": input9,
        "target": target9,
        "log_input": True,
        "full": False,
        "eps": 1e-6,
        "reduction": "sum"
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([1.0, 1.0, 1.0])
    target10 = np.array([1, 1, 1])
    input_dict10 = {
        "input": input10,
        "target": target10,
        "log_input": False,
        "full": True,
        "eps": 1e-8,
        "reduction": "mean"
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.poisson_nll_loss"] = poisson_nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.poisson_nll_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.poisson_nll_loss'.")


check_valid('torch.nn.functional.poisson_nll_loss', generated_inputs['torch.nn.functional.poisson_nll_loss'], lib="torch", suffix=0)
