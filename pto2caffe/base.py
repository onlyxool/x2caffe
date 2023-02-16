from abc import ABC
from enum import Enum


class Status(Enum):
    # Before `__init__()` finishes.
    UNINITIALIZED = 0

    # Basic Model objects registed, Class-wise member allocated.
    INITIALIZED = 1

    # Objects and any members have been parsed from model.
    PARSED = 2

    # Model object has been created.
    CONVERTED = 3

    # Forwarded
    FORWARDED = 4

    # Reserved.
    INVALID = 10

    @property
    def uninitialized(self):
        return self == self.UNINITIALIZED

    @property
    def initialized(self):
        return self == self.INITIALIZED

    @property
    def parsed(self):
        return self == self.PARSED

    @property
    def converted(self):
        return self == self.CONVERTED

    @property
    def forwarded(self):
        return self == self.FORWARDED


class Base(ABC):
    def __init__(self, model=None, graph=None, index=None):
        # Overall fields
        self.status = Status.UNINITIALIZED

        self.model = model
        self.graph = graph
        self.index = index  # index of tensor or op

        self.inputs = list()
        self.inputs_buf = list()
        self.inputs_shape = list()
        self.inputs_dtype = list()
        self.inputs_maxval = list()
        self.inputs_minval = list()

        self.outputs = list()
        self.outputs_shape = list()
        self.outputs_dtype = list()
        self.outputs_maxval = list()
        self.outputs_minval = list()

        # Caffe object
        self.caffe = None

    def setInited(self):
        assert(self.status.uninitialized)
        self.status = Status.INITIALIZED

    def parse(self):
        raise NotImplementedError("method parse() should be overrided!")

    def setParsed(self):
        assert(self.status.initialized)
        self.status = Status.PARSED

    def validate(self):
        raise NotImplementedError("method validate() should be overrided!")

    def convert(self):
        raise NotImplementedError("method convert() should be overrided!")

    def setConverted(self):
        assert(self.status.parsed)
        self.status = Status.CONVERTED

    def setForwarded(self):
        self.status = Status.FORWARDED

    def setInvalid(self):
        self.status = Status.INVALID

    @property
    def shorty(self):
        """A short readable description for the class/object.

        This aims to be different from `__str__` which is exepcted to be
        long description on this package.
        """
        raise NotImplementedError("method shorty() should be overrided!")

    def __str__(self):
        """A readable description for the class/object."""
        raise NotImplementedError("method __str__() should be overrided!")
