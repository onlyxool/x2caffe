import re


def get_relay_type(relay):
    type_str = relay[:relay.find('(%')].split(' ')[-1]
    if type_str == '':
        return 'array'
    else:
        return type_str


def get_tensor_shape(tytensor):
    return re.compile(r'Tensor\[\((.+?)\),').findall(tytensor)


def preprocess(relay):
    def remove(match):
        return match.group()[:match.start()-match.end()+1]
    relay = re.sub(r'\df', remove, relay)
    relay = re.sub(r'\di64', remove, relay)

    return relay.strip()
