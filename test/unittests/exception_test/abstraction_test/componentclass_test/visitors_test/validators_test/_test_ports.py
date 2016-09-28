import unittest
from nineml.abstraction.componentclass.visitors.validators.ports import (PortConnectionsComponentValidator)
from nineml.utils.testing.comprehensive import instances_of_all_types
from nineml.exceptions import (NineMLRuntimeError)


class TestPortConnectionsComponentValidatorExceptions(unittest.TestCase):

    def test___init___ninemlruntimeerror(self):
        """
        line #: 35
        message: err

        context:
        --------
    def __init__(self, component_class, **kwargs):  # @UnusedVariable
        BaseValidator.__init__(self, require_explicit_overrides=False)

        self.ports = defaultdict(list)
        self.portconnections = list()

        self.visit(component_class)

        connected_recv_ports = set()

        # Check for duplicate connections in the
        # portconnections. This can only really happen in the
        # case of connecting 'send to reduce ports' more than once.
        seen_port_connections = set()
        for pc in self.portconnections:
            if pc in seen_port_connections:
                err = 'Duplicate Port Connection: %s -> %s' % (pc[0], pc[1])
        """

        portconnectionscomponentvalidator = next(instances_of_all_types['PortConnectionsComponentValidator'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            portconnectionscomponentvalidator.__init__,
            component_class=None)

    def test___init___ninemlruntimeerror2(self):
        """
        line #: 42
        message: BinOp(left=Str(s='Unable to find port specified in connection: %s'), op=Mod(), right=Name(id='src', ctx=Load()))

        context:
        --------
    def __init__(self, component_class, **kwargs):  # @UnusedVariable
        BaseValidator.__init__(self, require_explicit_overrides=False)

        self.ports = defaultdict(list)
        self.portconnections = list()

        self.visit(component_class)

        connected_recv_ports = set()

        # Check for duplicate connections in the
        # portconnections. This can only really happen in the
        # case of connecting 'send to reduce ports' more than once.
        seen_port_connections = set()
        for pc in self.portconnections:
            if pc in seen_port_connections:
                err = 'Duplicate Port Connection: %s -> %s' % (pc[0], pc[1])
                raise NineMLRuntimeError(err)
            seen_port_connections.add(pc)

        # Check each source and sink exist,
        # and that each recv port is connected at max once.
        for src, sink in self.portconnections:
            if src not in self.ports:
        """

        portconnectionscomponentvalidator = next(instances_of_all_types['PortConnectionsComponentValidator'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            portconnectionscomponentvalidator.__init__,
            component_class=None)

    def test___init___ninemlruntimeerror3(self):
        """
        line #: 46
        message: BinOp(left=Str(s='Port was specified as a source, but is incoming: %s'), op=Mod(), right=Name(id='src', ctx=Load()))

        context:
        --------
    def __init__(self, component_class, **kwargs):  # @UnusedVariable
        BaseValidator.__init__(self, require_explicit_overrides=False)

        self.ports = defaultdict(list)
        self.portconnections = list()

        self.visit(component_class)

        connected_recv_ports = set()

        # Check for duplicate connections in the
        # portconnections. This can only really happen in the
        # case of connecting 'send to reduce ports' more than once.
        seen_port_connections = set()
        for pc in self.portconnections:
            if pc in seen_port_connections:
                err = 'Duplicate Port Connection: %s -> %s' % (pc[0], pc[1])
                raise NineMLRuntimeError(err)
            seen_port_connections.add(pc)

        # Check each source and sink exist,
        # and that each recv port is connected at max once.
        for src, sink in self.portconnections:
            if src not in self.ports:
                raise NineMLRuntimeError(
                    'Unable to find port specified in connection: %s' %
                    (src))
            if self.ports[src].is_incoming():
        """

        portconnectionscomponentvalidator = next(instances_of_all_types['PortConnectionsComponentValidator'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            portconnectionscomponentvalidator.__init__,
            component_class=None)

    def test___init___ninemlruntimeerror4(self):
        """
        line #: 51
        message: BinOp(left=Str(s='Unable to find port specified in connection: %s'), op=Mod(), right=Name(id='sink', ctx=Load()))

        context:
        --------
    def __init__(self, component_class, **kwargs):  # @UnusedVariable
        BaseValidator.__init__(self, require_explicit_overrides=False)

        self.ports = defaultdict(list)
        self.portconnections = list()

        self.visit(component_class)

        connected_recv_ports = set()

        # Check for duplicate connections in the
        # portconnections. This can only really happen in the
        # case of connecting 'send to reduce ports' more than once.
        seen_port_connections = set()
        for pc in self.portconnections:
            if pc in seen_port_connections:
                err = 'Duplicate Port Connection: %s -> %s' % (pc[0], pc[1])
                raise NineMLRuntimeError(err)
            seen_port_connections.add(pc)

        # Check each source and sink exist,
        # and that each recv port is connected at max once.
        for src, sink in self.portconnections:
            if src not in self.ports:
                raise NineMLRuntimeError(
                    'Unable to find port specified in connection: %s' %
                    (src))
            if self.ports[src].is_incoming():
                raise NineMLRuntimeError(
                    'Port was specified as a source, but is incoming: %s' %
                    (src))

            if sink not in self.ports:
        """

        portconnectionscomponentvalidator = next(instances_of_all_types['PortConnectionsComponentValidator'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            portconnectionscomponentvalidator.__init__,
            component_class=None)

    def test___init___ninemlruntimeerror5(self):
        """
        line #: 56
        message: BinOp(left=Str(s='Port was specified as a sink, but is not incoming: %s'), op=Mod(), right=Name(id='sink', ctx=Load()))

        context:
        --------
    def __init__(self, component_class, **kwargs):  # @UnusedVariable
        BaseValidator.__init__(self, require_explicit_overrides=False)

        self.ports = defaultdict(list)
        self.portconnections = list()

        self.visit(component_class)

        connected_recv_ports = set()

        # Check for duplicate connections in the
        # portconnections. This can only really happen in the
        # case of connecting 'send to reduce ports' more than once.
        seen_port_connections = set()
        for pc in self.portconnections:
            if pc in seen_port_connections:
                err = 'Duplicate Port Connection: %s -> %s' % (pc[0], pc[1])
                raise NineMLRuntimeError(err)
            seen_port_connections.add(pc)

        # Check each source and sink exist,
        # and that each recv port is connected at max once.
        for src, sink in self.portconnections:
            if src not in self.ports:
                raise NineMLRuntimeError(
                    'Unable to find port specified in connection: %s' %
                    (src))
            if self.ports[src].is_incoming():
                raise NineMLRuntimeError(
                    'Port was specified as a source, but is incoming: %s' %
                    (src))

            if sink not in self.ports:
                raise NineMLRuntimeError(
                    'Unable to find port specified in connection: %s' %
                    (sink))

            if not self.ports[sink].is_incoming():
        """

        portconnectionscomponentvalidator = next(instances_of_all_types['PortConnectionsComponentValidator'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            portconnectionscomponentvalidator.__init__,
            component_class=None)

    def test___init___ninemlruntimeerror6(self):
        """
        line #: 62
        message: BinOp(left=Str(s="Port was 'recv' and specified twice: %s"), op=Mod(), right=Name(id='sink', ctx=Load()))

        context:
        --------
    def __init__(self, component_class, **kwargs):  # @UnusedVariable
        BaseValidator.__init__(self, require_explicit_overrides=False)

        self.ports = defaultdict(list)
        self.portconnections = list()

        self.visit(component_class)

        connected_recv_ports = set()

        # Check for duplicate connections in the
        # portconnections. This can only really happen in the
        # case of connecting 'send to reduce ports' more than once.
        seen_port_connections = set()
        for pc in self.portconnections:
            if pc in seen_port_connections:
                err = 'Duplicate Port Connection: %s -> %s' % (pc[0], pc[1])
                raise NineMLRuntimeError(err)
            seen_port_connections.add(pc)

        # Check each source and sink exist,
        # and that each recv port is connected at max once.
        for src, sink in self.portconnections:
            if src not in self.ports:
                raise NineMLRuntimeError(
                    'Unable to find port specified in connection: %s' %
                    (src))
            if self.ports[src].is_incoming():
                raise NineMLRuntimeError(
                    'Port was specified as a source, but is incoming: %s' %
                    (src))

            if sink not in self.ports:
                raise NineMLRuntimeError(
                    'Unable to find port specified in connection: %s' %
                    (sink))

            if not self.ports[sink].is_incoming():
                raise NineMLRuntimeError(
                    'Port was specified as a sink, but is not incoming: %s' %
                    (sink))

            if self.ports[sink].mode == 'recv':
                if self.ports[sink] in connected_recv_ports:
        """

        portconnectionscomponentvalidator = next(instances_of_all_types['PortConnectionsComponentValidator'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            portconnectionscomponentvalidator.__init__,
            component_class=None)

    def test__action_port_ninemlruntimeerror(self):
        """
        line #: 68
        message: BinOp(left=Str(s='Duplicated Name for port found: %s'), op=Mod(), right=Attribute(value=Name(id='port', ctx=Load()), attr='name', ctx=Load()))

        context:
        --------
    def _action_port(self, port):
        if port.name in self.ports:
        """

        portconnectionscomponentvalidator = next(instances_of_all_types['PortConnectionsComponentValidator'].itervalues())
        self.assertRaises(
            NineMLRuntimeError,
            portconnectionscomponentvalidator._action_port,
            port=None)
