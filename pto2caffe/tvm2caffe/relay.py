import re


def get_relay_type(relay):
    return relay[:relay.find('(')].split(' ')[-1]


def get_tensor_shape(tytensor):
    return re.compile(r'Tensor\[\((.+?)\),').findall(tytensor)


def preprocess(relay):
    def remove(match):
        return match.group()[:match.start()-match.end()+1]
    relay = re.sub(r'\df', remove, relay)
    relay = re.sub(r'\di64', remove, relay)
    relay = relay.replace('(meta', '( meta')

    return relay.strip()
