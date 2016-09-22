import unittest
from nineml.abstraction.ports import (AnalogReducePort)
from nineml.utils.testing.comprehensive import instances_of_all_types
from nineml.exceptions import (NineMLRuntimeError)


class TestAnalogReducePortExceptions(unittest.TestCase):

    def test___init___ninemlruntimeerror(self):
        """
        line #: 231
        message: err

        context:
        --------
    def __init__(self, name, dimension=None, operator='+'):
        if operator not in self._operator_map.keys():
            err = ("%s('%s')" + "specified undefined operator: '%s'") %\
                  (self.__class__.__name__, name, str(operator))
        """

        analogreduceport = next(instances_of_all_types['AnalogReducePort'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            analogreduceport.__init__,
            name=None,
            dimension=None,
            operator='+')

