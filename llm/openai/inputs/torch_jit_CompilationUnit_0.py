
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def compilation_unit_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2, 3, 4).numpy()
    list_of_inputs.append({"torch.jit.CompilationUnit": input1})
    
    input2 = torch.zeros(5, 5).numpy()
    list_of_inputs.append({"torch.jit.CompilationUnit": input2})
    
    input3 = torch.ones(1, 2, 3, 4).numpy()
    list_of_inputs.append({"torch.jit.CompilationUnit": input3})
    
    input4 = np.random.rand(3, 2)
    list_of_inputs.append({"torch.jit.CompilationUnit": input4})
    
    input5 = np.zeros((4, 4, 4))
    list_of_inputs.append({"torch.jit.CompilationUnit": input5})

    input6 = np.ones((2, 2, 2, 2))
    list_of_inputs.append({"torch.jit.CompilationUnit": input6})

    input7 = torch.randint(0, 10, (2, 3)).numpy()
    list_of_inputs.append({"torch.jit.CompilationUnit": input7})
    
    input8 = torch.randn(3, 3, 3).numpy() * -1
    list_of_inputs.append({"torch.jit.CompilationUnit": input8})

    input9 = np.random.rand(10)
    list_of_inputs.append({"torch.jit.CompilationUnit": input9})
    
    input10 = np.zeros((2, 3, 1))
    list_of_inputs.append({"torch.jit.CompilationUnit": input10})
    
    input11 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    list_of_inputs.append({"torch.jit.CompilationUnit": input11})
    
    return list_of_inputs

generated_inputs["torch.jit.CompilationUnit"] = compilation_unit_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.CompilationUnit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.CompilationUnit'.")


check_valid('torch.jit.CompilationUnit', generated_inputs['torch.jit.CompilationUnit'], lib="torch", suffix=0)
