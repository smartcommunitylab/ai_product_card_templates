import yaml
from fairness.holisticAI.src.data_preparation import *
from fairness.holisticAI.src.modelling import *

def main():
    with open("fairness/holisticAI/metadata/aipc.yaml", "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file) 
        # operations
        operations = aipc_configs["operations"]
        data_operations = [elem for elem in operations if elem["type"] == "data_preprocessing"]
        model_operations = [elem for elem in operations if elem["type"] == "model_training"]
        # artifacts
        artifacts = aipc_configs["artifacts"]
        data_artifacts = artifacts["data"]
        model_artifacts = artifacts["model"]
        configuration_artifacts = artifacts["configuration"]
        # run data operations
        run_data_operations(data_operations, data_artifacts, configuration_artifacts)
        # run modelling operation
        run_modelling_operations(model_operations, model_artifacts, data_artifacts, configuration_artifacts)
        
        
        
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
                artifact_vars = {var_name: var_value
                            for var_name, var_value in input_artifact[0].items() 
                            if var_name != "name"}
                input_vars.update({"data": Data(**artifact_vars)})
            input_artifact = [art for art in config_artifacts if art["name"] == input_name[0]]
            if len(input_artifact) > 0:
                artifact_vars = {var_name: var_value
                            for var_name, var_value in input_artifact[0].items() 
                            if var_name != "name"}
                input_vars.update({"config": Configuration(**artifact_vars)})
        print(input_vars)
    
        method_name = globals()[method_name]
        result = method_name(**input_vars)
        

def run_modelling_operations(model_operations, model_artifacts, data_artifacts, config_artifacts):
    for operation in model_operations:
        specs = operation["implementation"]["spec"]
        method_name = specs["method_name"]
        inputs = specs["inputs"]
        outputs = specs["outputs"]
        input_vars = {}
        print(inputs)
        for my_input in inputs:            
            input_name = list(my_input.values())
            input_artifact = [art for art in data_artifacts if art["name"] == input_name[0]]
            if len(input_artifact) > 0:
                artifact_vars = {var_name: var_value
                            for var_name, var_value in input_artifact[0].items() 
                            if var_name != "name"}
                input_vars.update({"data": Data(**artifact_vars)})
            input_artifact = [art for art in config_artifacts if art["name"] == input_name[0]]
            if len(input_artifact) > 0:
                artifact_vars = {var_name: var_value
                            for var_name, var_value in input_artifact[0].items() 
                            if var_name != "name"}
                input_vars.update({"config": Configuration(**artifact_vars)})

        print(input_vars)
    
        method_name = globals()[method_name]
        result = method_name(**input_vars)

def test_coverage_dimensions(dimension: list[str] = "fairness"):
    # Give feedback on missing implementation aspects
    # Automate Reporting: Demonstrate that the AI system complies with the requirements 
    # Automate Enforcement: Guardrails/Test coverage for requirements, such as 
    # [fairness, robustness, accuracy, optimization, transparency, privacy, explainability]
    pass

if __name__=="__main__":
    main()