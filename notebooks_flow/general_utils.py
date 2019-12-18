import ast
import re


EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")


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


def validate_telefono_num(telefono):

    if len(telefono) == 9 and telefono.startswith('9'):
        return True
    elif len(telefono) == 6:
        return True
    elif len(telefono) == 7:
        return True
    else:
        return False


def validate_dni_num(dni):
    if len(dni) == 8:
        return True
    else:
        return False


def validate_mail(email):
    if EMAIL_REGEX.fullmatch(email):
        return True
    else:
        return False

