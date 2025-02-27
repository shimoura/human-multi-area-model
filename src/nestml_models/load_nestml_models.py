import os
# from pynestml.codegeneration.nest_code_generator_utils import NESTCodeGeneratorUtils
from pynestml.frontend.pynestml_frontend import generate_target

def generate_nestml_model(model_name):
    """
    Generates NESTML model code if not already generated.

    Parameters
    ----------
    model_name : str
        Name of the NESTML model to generate.

    Returns
    -------
    module_name : str
        Name of the generated module.
    """
    # Check if the model is available
    if model_name == "iaf_psc_exp_multisyn_exc_neuron":

        # Define the path to the nestml file
        path_to_models = "src/nestml_models"
        nestml_model_dir = os.path.join(path_to_models, model_name + ".nestml")

        # Path to the generated code model
        path_to_generated_models = os.path.join(path_to_models, "generated_nestml_models")

        # Define the module name
        module_name = "iaf_psc_exp_multisyn_exc_module"

        # Check if the code generation was already performed
        if not os.path.exists(os.path.join(path_to_generated_models, module_name + ".h")):
            # Generate the code
            generate_target(input_path = nestml_model_dir,
                            target_platform = "NEST",
                            target_path = path_to_generated_models,
                            module_name = module_name)
        else:
            print("NESTML model " + model_name + " already generated")

    # If the model is not available
    else:
        raise ValueError(f"Model {model_name} not found")

    return module_name