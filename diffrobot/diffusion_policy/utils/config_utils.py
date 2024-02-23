
import importlib.util
import os
import sys


def get_config(config_file):
    config_file += '.py'
    config_file = os.path.join("configs", config_file)
    module_name = os.path.basename(config_file).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module