import re


def get_relay_type(relay):
#    return re.compile(r'%[0-9]+ = ([0-9a-zA-Z\._]+?)\(').findall(relay)[0]

    type_str = re.compile(r'%.+? = (.+?)\%').findall(relay)
    if len(type_str) == 1 and type_str[0] == '(':
        return 'array'
    elif len(type_str) == 1:
        return type_str[0][:-1]
    elif len(type_str) == 0:
        return 'bypass'


def get_tensor_shape(tytensor):
    return re.compile(r'Tensor\[\((.+?)\),').findall(tytensor)


def preprocess(relay):
    def remove(match):
        return match.group()[:match.start()-match.end()+1]
    relay = re.sub(r'\df', remove, relay)
    relay = re.sub(r'\di64', remove, relay)

    return relay
