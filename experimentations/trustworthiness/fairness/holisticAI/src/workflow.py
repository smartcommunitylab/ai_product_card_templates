import yaml
from data_preparation import *
from modelling import *

def main():
    with open("metadata/aipc.yaml", "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file) 
        # operations
        operations = aipc_configs["operations"]
        data_operations = [elem for elem in operations if elem["type"] == "data_preprocessing"]
        model_operations = [elem for elem in operations if elem["type"] == "modelling"]
        # artifacts
        artifacts = aipc_configs["artifacts"]
        data_artifacts = artifacts["data"]
        model_artifacts = artifacts["model"]
        configuration_artifacts = artifacts["configuration"]
        # run operations
        run_data_operations(data_operations, data_artifacts, configuration_artifacts)
        
def run_data_operations(data_operations, data_artifacts, config_artifacts):
    for operation in data_operations:
        specs = operation["implementation"]["spec"]
        method_name = specs["method_name"]
        inputs = specs["inputs"]
        outputs = specs["outputs"]
        input_vars = {}
        for my_input in inputs:            
            input_name = list(my_input.values())
            input_artifact = [art for art in data_artifacts if art["name"] == input_name[0]]
            if len(input_artifact) > 0:
                input_vars.update({var_name: var_value
                            for var_name, var_value in input_artifact[0].items() 
                            if var_name != "name"})
            input_artifact = [art for art in config_artifacts if art["name"] == input_name[0]]
            if len(input_artifact) > 0:
                input_vars.update({var_name: var_value
                            for var_name, var_value in input_artifact[0].items() 
                            if var_name != "name"})
        print(input_vars)
    
        method_name = globals()[method_name]
        result = method_name(**input_vars)
        

def run_modelling_operations(model_operations, model_artifacts):
    pass

if __name__=="__main__":
    main()