
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def get_file_path_inputs():
    list_of_inputs = []
    
    input1 = "https://example.com/model.pth"
    input_dict1 = {"url": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = "http://www.example.com/data.txt"
    input_dict2 = {"url": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = "https://storage.googleapis.com/my-bucket/my-file.bin"
    input_dict3 = {"url": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = "ftp://ftp.example.com/public/data.zip"
    input_dict4 = {"url": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = "s3://my-bucket/my-object.csv"
    input_dict5 = {"url": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = "https://www.example.com/model_with_long_name.pth"
    input_dict6 = {"url": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = "https://example.com/file.tar.gz"
    input_dict7 = {"url": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = "https://example.com/file_with_query_params.txt?version=1.0"
    input_dict8 = {"url": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = "https://example.com/file_with_fragment.html#section1"
    input_dict9 = {"url": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = "https://example.com/a/b/c/d/e/file.jpg"
    input_dict10 = {"url": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.get_file_path"] = get_file_path_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.get_file_path' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_file_path'.")


check_valid('torch.get_file_path', generated_inputs['torch.get_file_path'], lib="torch", suffix=0)
