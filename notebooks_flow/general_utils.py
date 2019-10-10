import ast


def get_conf_file_of_camp(path, id_camp):
    # '../params/campaigns'
    a = None
    with open(path, 'r') as f:
        s = f.read()
        a = ast.literal_eval(s)

    if a is not None:
        return a[str(id_camp)]['conf_file']
    else:
        return None


def get_param_from_file(path, param):
    a = None
    with open(path, 'r') as f:
        s = f.read()
        a = ast.literal_eval(s)

    if a is not None:
        return a[str(param)]
    else:
        return None
