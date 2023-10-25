import re


def get_relay_type(relay):
    return relay[:relay.find('(')].split(' ')[-1]


def get_tensor_shape(tytensor):
    return re.compile(r'Tensor\[\((.+?)\),').findall(tytensor)


def remove_numTypeExt(relay):
    def remove(match):
        return match.group()[:match.start()-match.end()+1]

    relay = re.sub(r'\df', remove, relay)
    relay = re.sub(r'\di64', remove, relay)
    relay = relay.replace('meta', '$meta')

    return relay


def preprocess(relay):
    relay = remove_numTypeExt(relay)

    first_operand = re.compile(r'\(([-+]?\d*\.\d+|\d+) \/\* ty=').findall(relay.split(get_relay_type(relay))[1].split(') /*')[0])
    if len(first_operand) == 1:
        relay = relay.replace(get_relay_type(relay)+'('+first_operand[0], get_relay_type(relay)+'($'+first_operand[0], 1)

    first_operand = re.compile(r'\((\d+[e|E][+|-]\d+) \/\* ty=').findall(relay.split(get_relay_type(relay))[1].split(') /*')[0])
    if len(first_operand) == 1:
        relay = relay.replace(get_relay_type(relay)+'('+first_operand[0], get_relay_type(relay)+'($'+first_operand[0], 1)

    return relay.strip()
