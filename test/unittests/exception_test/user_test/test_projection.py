import unittest
from nineml.user.projection import (BaseConnectivity)
from nineml.utils.testing.comprehensive import instances_of_all_types
from nineml.exceptions import (NineMLRuntimeError)


class TestBaseConnectivityExceptions(unittest.TestCase):

    def test___init___ninemlruntimeerror(self):
        """
        line #: 37
        message: Cannot connect to populations of different sizes ({} and {}) with OneToOne connection rule

        context:
        --------
    def __init__(self, connection_rule_properties, source_size,
                 destination_size,
                 **kwargs):  # @UnusedVariable
        if (connection_rule_properties.lib_type == 'OneToOne' and
                source_size != destination_size):
        """

        baseconnectivity = instances_of_all_types['BaseConnectivity']
        self.assertRaises(
            NineMLRuntimeError,
            baseconnectivity.__init__,
            connection_rule_properties=None,
            source_size=None,
            destination_size=None)

