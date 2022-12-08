import yaml
import os


def open_file(mode):
    yml_path = os.path.dirname(__file__) + '\\model_tmp.yaml'
    assert os.path.exists(yml_path), 'yaml file doesnt exist'
    f = open(yml_path, mode)
    return f


def yml_parse2():
    f = open_file('r')
    parsed = yaml.safe_load(f)
    f.close()
    return parsed


def yml_save(key, name):
    parse = yml_parse2()
    parse[key] = name
    f = open_file('w')
    yaml.dump(parse, f)
    f.close()
    return
