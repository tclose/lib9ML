from itertools import product
from abc import ABCMeta, abstractmethod
from . import BaseULObject
from nineml.abstraction.connectionrule import (
    explicit_connection_rule, one_to_one_connection_rule)
from nineml.user.port_connections import EventPortConnection
from nineml.user.connectionrule import (
    ConnectionRuleProperties)
from nineml.units import Quantity
from nineml.abstraction.ports import (
    SendPort, ReceivePort, EventPort, AnalogPort, Port)
from nineml.user.component_array import ComponentArray
from nineml.base import DocumentLevelObject
from future.utils import with_metaclass
from nineml.utils import validate_identifier


class BaseConnectionGroup(
        with_metaclass(ABCMeta,
                       type('NewBase',
                            (BaseULObject, DocumentLevelObject), {}))):

    nineml_attr = ('name', 'source_port', 'destination_port')
    nineml_child = {'source': ComponentArray,
                    'destination': ComponentArray,
                    'connectivity': ConnectionRuleProperties,
                    'delay': Quantity}

    def __init__(self, name, source, destination, source_port,
                 destination_port, delay, connectivity):
        self._name = validate_identifier(name)
        BaseULObject.__init__(self)
        DocumentLevelObject.__init__(self)
        self._source = source
        self._destination = destination
        self._source_port = source_port
        self._destination_port = destination_port
        self._connectivity = connectivity
        self._delay = delay
        if isinstance(source_port, Port):
            self._check_ports(source_port, destination_port)

    @property
    def name(self):
        return self._name

    @property
    def source(self):
        return self._source

    @property
    def destination(self):
        return self._destination

    @property
    def source_port(self):
        return self._source_port

    @property
    def connectivity(self):
        return self._connectivity

    @property
    def destination_port(self):
        return self._destination_port

    @property
    def delay(self):
        return self._delay

    def __len__(self):
        return len(self.connections)

    @classmethod
    def from_port_connection(self, port_conn, projection, component_arrays,
                             connections):
        if isinstance(port_conn, EventPortConnection):
            cls = EventConnectionGroup
        else:
            cls = AnalogConnectionGroup
        num_connections = len(connections)
        name = '__'.join((
            projection.name, port_conn.sender_role,
            port_conn.send_port_name, port_conn.receiver_role,
            port_conn.receive_port_name))
        if (port_conn.sender_role in ('response', 'plasticity') and
                port_conn.receiver_role in ('response', 'plasticity')):
            # This should be able to be dropped after move to merged synapse
            # components in v2
            conn_props = ConnectionRuleProperties(
                name=name + '_connectivity',
                definition=one_to_one_connection_rule)
            return [cls(
                name,
                component_arrays[projection.name +
                                 ComponentArray.suffix[port_conn.sender_role]],
                component_arrays[
                    projection.name +
                    ComponentArray.suffix[port_conn.receiver_role]],
                source_port=port_conn.send_port_name,
                destination_port=port_conn.receive_port_name,
                connectivity=conn_props, delay=None)]
        else:
            if (port_conn.sender_role == 'pre' and
                    port_conn.receiver_role == 'post'):
                conns = connections
            elif (port_conn.sender_role == 'post' and
                  port_conn.receiver_role == 'pre'):
                conns = connections.inverse()
            elif port_conn.sender_role == 'pre':
                conns = [
                    (s, i) for i, (s, _) in enumerate(sorted(connections))]
            elif port_conn.receiver_role == 'post':
                conns = [
                    (i, d) for i, (_, d) in enumerate(sorted(connections))]
            else:
                assert False
            # FIXME: This will need to change in version 2, when each connection
            #        has its own delay
            if port_conn.sender_role == 'pre':
                delay = projection.delay
            else:
                delay = None
            # Get source and destination component arrays
            if port_conn.sender_role == 'pre':
                source_pop = projection.pre
                source_len = len(source_pop)
            elif port_conn.sender_role == 'post':
                source_pop = projection.post
                source_len = len(source_pop)
            else:
                source_pop = projection  # The source comp-array is from syn.
                source_len = num_connections
            if port_conn.receiver_role == 'pre':
                dest_pop = projection.pre
                dest_len = len(dest_pop)
            elif port_conn.receiver_role == 'post':
                dest_pop = projection.post
                dest_len = len(dest_pop)
            else:
                dest_pop = projection
                dest_len = num_connections
            # Divide up selections into individual populations
            if source_pop.nineml_type == 'Selection':
                sources = []
                start_i = 0
                for pop in source_pop.populations:
                    end_i = start_i + pop.size
                    sources.append((pop, start_i, end_i))
                    start_i = end_i
            else:
                sources = [(source_pop, 0, source_len)]
            if dest_pop.nineml_type == 'Selection':
                destinations = []
                start_i = 0
                for pop in dest_pop.populations:
                    end_i = start_i + pop.size
                    destinations.append((pop, start_i, end_i))
                    start_i = end_i
            else:
                destinations = [(dest_pop, 0, dest_len)]
            # Return separate connection groups between all combinations of
            # source and destination populations/synapses
            conn_groups = []
            for i, ((source_pop, src_start, src_end),
                    (dest_pop, dest_start, dest_end)) in enumerate(product(
                        sources, destinations)):
                # Get source and destination component arrays
                source_array = component_arrays[
                    source_pop.name +
                    ComponentArray.suffix[port_conn.sender_role]]
                dest_array = component_arrays[
                    dest_pop.name +
                    ComponentArray.suffix[port_conn.receiver_role]]
                if len(sources) == 1 and len(destinations) == 1:
                    # There is only one conn_group generated from this port
                    # connection
                    conn_group_name = name
                    conn_group_conns = conns
                else:
                    conn_group_name = name + str(i)
                    # Determine connections that are relevant for the arrays
                    # in this connection group
                    conn_group_conns = [
                        (s - src_start, d - dest_start) for s, d in conns
                        if (s >= src_start and s < src_end and
                            d >= dest_start and d < dest_end)]
                if conn_group_conns:
                    source_inds, dest_inds = zip(*conn_group_conns)
                else:
                    source_inds, dest_inds = ()
                conn_props = ConnectionRuleProperties(
                    name=conn_group_name + '_connectivity',
                    definition=explicit_connection_rule,
                    properties={'sourceIndices': source_inds,
                                'destinationIndices': dest_inds})
                conn_groups.append(
                    cls(conn_group_name, source_array, dest_array,
                        source_port=port_conn.send_port_name,
                        destination_port=port_conn.receive_port_name,
                        connectivity=conn_props, delay=delay))
            return conn_groups

    @abstractmethod
    def _check_ports(self, source_port, destination_port):
        assert isinstance(source_port, SendPort)
        assert isinstance(destination_port, ReceivePort)

    def serialize_node(self, node, **options):  # @UnusedVariable
        source_elem = node.child(self.source, within='Source', **options)
        node.visitor.set_attr(source_elem, 'port', self.source_port,
                              **options)
        dest_elem = node.child(self.destination, within='Destination',
                               **options)
        node.visitor.set_attr(dest_elem, 'port', self.destination_port)
        node.child(self.connectivity, within='Connectivity')
        if self.delay is not None:
            node.child(self.delay, within='Delay')
        node.attr('name', self.name)

    @classmethod
    def unserialize_node(cls, node, **options):  # @UnusedVariable
        # Get Name
        name = node.attr('name', **options)
        connectivity = node.child(
            ConnectionRuleProperties, within='Connectivity', allow_ref=True,
            **options)
        source = node.child(ComponentArray, within='Source',
                            allow_ref='only', allow_within_attrs=True,
                            **options)
        destination = node.child(ComponentArray, within='Destination',
                                 allow_ref='only', allow_within_attrs=True,
                                 **options)
        source_elem = node.visitor.get_child(
            node.serial_element, 'Source', **options)
        source_port = node.visitor.get_attr(source_elem, 'port', **options)
        dest_elem = node.visitor.get_child(
            node.serial_element, 'Destination', **options)
        destination_port = node.visitor.get_attr(dest_elem, 'port', **options)
        delay = node.child(Quantity, within='Delay', allow_none=True,
                           **options)
        return cls(name=name, source=source, destination=destination,
                   source_port=source_port, destination_port=destination_port,
                   connectivity=connectivity, delay=delay)


class AnalogConnectionGroup(BaseConnectionGroup):

    nineml_type = 'AnalogConnectionGroup'
    communicates = 'analog'

    def _check_ports(self, source_port, destination_port):
        super(AnalogConnectionGroup, self)._check_ports(source_port,
                                                        destination_port)
        assert isinstance(source_port, AnalogPort)
        assert isinstance(destination_port, AnalogPort)


class EventConnectionGroup(BaseConnectionGroup):

    nineml_type = 'EventConnectionGroup'
    communicates = 'event'

    def _check_ports(self, source_port, destination_port):
        super(EventConnectionGroup, self)._check_ports(source_port,
                                                       destination_port)
        assert isinstance(source_port, EventPort)
        assert isinstance(destination_port, EventPort)
