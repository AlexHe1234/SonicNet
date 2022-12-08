import yaml
import os


def open_file(mode):
    yml_path = os.path.dirname(__file__) + '\\config.yaml'
    assert os.path.exists(yml_path), 'yaml file doesnt exist'
    f = open(yml_path, mode)
    return f


def yml_parse():
    f = open_file('r')
    parsed = yaml.safe_load(f)
    if parsed['uvo'] == 'none':
        parsed['uvo'] = None
    f.close()
    return parsed
