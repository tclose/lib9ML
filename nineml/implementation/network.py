from operator import itemgetter
from collections import defaultdict
from itertools import chain, repeat
from copy import copy
import networkx as nx
import nineml.units as un
from pprint import pprint
from nineml.user import (
    MultiDynamicsProperties, AnalogPortConnection, EventPortConnection,
    BasePortExposure)
from nineml.units import Quantity
from .dynamics import Dynamics, DynamicsClass
from .experiment import AnalogSource, EventSource, AnalogSink, EventSink
from .utils import create_progress_bar
from nineml.exceptions import NineMLUsageError


class Network(object):
    """
    An implementation of Network model in pure python. All populations and
    projections in the network model are first flattened to component-arrays
    and connection groups, then a network graph is constructed using networkx.
    Sub-graphs of nodes connected by delayless connections are merged into
    single nodes using MultiDynamics classes so their time derivatives can
    be solved simultaneously.

    Sources and sinks are included into the graph to play into and record from
    selected ports of selected components.

    Parameters
    ----------
    model : nineml.user.Network
        The model of the network described in 9ML
    start_t : Quantity(time)
        The starting time of the simulation
    sources : list(tuple(str, str, int, numpy.array))
        A list of explicit input signals to connect to the network. Each tuple
        consists of 4 values corresponding to,
            <comp-array-name>, <receive-port-name>, <node-index>, <signal>
    sinks : list(tuple(str, str[, array(int)]))
        A list of sinks to add to the network in order to save activity. Each
        tuple consists of 2 or 3 values corresponding to:
            <comp-array-name>, <send-port-name>, <node-indices> (optional)
        If node indices are not provided then sinks are added to all nodes
    """

    def __init__(self, model, start_t, sources=None, sinks=None):
        if isinstance(start_t, Quantity):
            start_t = float(start_t.in_si_units())
        if sources is None:
            sources = []
        if sinks is None:
            sinks = []
        self.t = start_t
        component_arrays, connection_groups = model.flatten()
        # Initialise a graph to represent the network
        self.graph = nx.MultiDiGraph()
        # Add nodes (2-tuples consisting of <component-array-name> and
        # <cell-index>) for each component in each array
        ca_dict = {}
        for comp_array in component_arrays:
            ca_dict[comp_array.name] = comp_array
            props = comp_array.dynamics_properties
            for i in range(comp_array.size):
                self.graph.add_node((comp_array.name, i),
                                    properties=props.sample(i))
        # Add connections between components from connection groups
        self.min_delay = float('inf')
        for conn_group in connection_groups:
            delay_qty = conn_group.delay
            if delay_qty is None:
                delays = repeat(0.0)
            else:
                delays = (float(d) for d in delay_qty.in_si_units())
            for (src_i, dest_i), delay in zip(conn_group.connections, delays):
                self.graph.add_edge(
                    (conn_group.source.name, int(src_i)),
                    (conn_group.destination.name, int(dest_i)),
                    communicates=conn_group.communicates,
                    delay=delay,
                    src_port=conn_group.source_port,
                    dest_port=conn_group.destination_port)
                if delay and delay < self.min_delay:
                    self.min_delay = delay
        # Add sources to network graph
        self.sources = defaultdict(list)
        for comp_array_name, port_name, index, signal in sources:
            port = ca_dict[comp_array_name].dynamics_properties.port[port_name]
            if port.communicates == 'analog':
                source_cls = AnalogSource
            else:
                source_cls = EventSource
            source_array_name = '{}_{}' .format(comp_array_name, port_name)
            source = source_cls(source_array_name + str(index), signal)
            self.sources[source_array_name].append(source)
            # NB: Use negative index to avoid any (unlikely) name-clashes
            #     with other component arrays
            self.graph.add_node((source_array_name, -(index + 1)),
                                source=source)
            self.graph.add_edge(
                (source_array_name, -(index + 1)),
                (comp_array_name, index),
                communicates=port.communicates,
                delay=self.min_delay,
                dest_port=port_name)
        # Add sinks to network graph
        self.sinks = defaultdict(list)
        for sink_tuple in sinks:
            try:
                comp_array_name, port_name, indices = sink_tuple
            except ValueError:
                comp_array_name, port_name = sink_tuple
                try:
                    indices = range(ca_dict[comp_array_name].size)
                except KeyError:
                    raise NineMLUsageError(
                        "Unrecognised component array name in list of sinks "
                        "'{}'".format(comp_array_name))
            port = ca_dict[comp_array_name].dynamics_properties.port(port_name)
            if port.communicates == 'analog':
                sink_cls = AnalogSink
            else:
                sink_cls = EventSink
            sink_array_name = '{}_{}' .format(comp_array_name, port_name)
            for index in indices:
                sink = sink_cls(sink_array_name + str(index))
                self.sinks[sink_array_name].append(sink)
                # NB: Use negative index to avoid any (unlikely) name-clashes
                #     with other component arrays
                self.graph.add_node((sink_array_name, -(index + 1)), sink=sink)
                self.graph.add_edge(
                    (comp_array_name, index),
                    (sink_array_name, -(index + 1)),
                    communicates=port.communicates,
                    delay=self.min_delay,
                    src_port=port_name)
        # Merge dynamics definitions for nodes connected without delay
        # connections into multi-dynamics definitions. We save the iterator
        # into a list as we will be removing nodes as they are merged.
        for node in list(self.graph.nodes):
            if node not in self.graph:
                continue  # If node has already been merged
            conn_without_delay = self.connected_without_delay(node)
            if conn_without_delay:
                self.merge_nodes(conn_without_delay)
        # Initialise all dynamics components in graph
        dyn_class_cache = []  # Cache for storing previously analysed classes
        self.components = []
        for node, attr in self.graph.nodes(data=True):
            # Attempt to reuse DynamicsClass objects between Dynamics objects
            # to save reanalysing their equations
            try:
                model = attr['properties'].component_class
            except KeyError:
                continue  # Source and sink nodes have already been initialised
            try:
                dyn_class = next(dc for m, dc in dyn_class_cache if m == model)
            except StopIteration:
                dyn_class = DynamicsClass(model)
                dyn_class_cache.append((model, dyn_class))
            # Create dynamics object
            attr['dynamics'] = dynamics = Dynamics(
                attr['properties'], start_t, dynamics_class=dyn_class,
                name='{}_{}'.format(*node))
            self.components.append(dynamics)
        # Make all connections between dynamics components, sources and sinks
        for u, v, conn in self.graph.out_edges(data=True):
            u_attr = self.graph.nodes[u]
            v_attr = self.graph.nodes[v]
            try:
                from_port = u_attr['dynamics'].ports[conn['src_port']]
            except KeyError:
                from_port = u_attr['source']
            try:
                to_port = v_attr['dynamics'].ports[conn['dest_port']]
            except KeyError:
                to_port = v_attr['sink']
            from_port.connect_to(to_port, delay=conn['delay'])

    def simulate(self, stop_t, dt, progress_bar=True):
        if isinstance(stop_t, Quantity):
            stop_t = float(stop_t.in_units(un.s))
        if isinstance(dt, Quantity):
            dt = float(dt.in_si_units())
        if progress_bar is True:
            progress_bar = create_progress_bar(self.t, stop_t, self.min_delay)
            progress_bar.update(self.t)
        while self.t < stop_t:
            self.t = min(stop_t, self.t + self.min_delay)
            for component in self.components:
                component.simulate(self.t, dt, progress_bar=False)
            if progress_bar:
                progress_bar.update(self.t)
        if progress_bar is not None:
            progress_bar.finish()

    def connected_without_delay(self, start_node, connected=None):
        """
        Returns the sub-graph of nodes connected to the start node by a chain
        of delayless connections.

        Parameters
        ----------
        start_node : tuple(str, int)
            The starting node to check for delayless connected neighbours from
        connected : set(tuple(str, int))
            The set eventually returned by the method, passed as a arg to
            recursive calls of this method
        """
        if connected is None:
            connected = set()
        # Iterate all in-coming and out-going edges and check for
        # any zero delays. If so, add to set of nodes to merge
        for neigh, conn in chain(
            ((n, c) for n, _, c in self.graph.in_edges(start_node,
                                                       data=True)),
            ((n, c) for _, n, c in self.graph.out_edges(start_node,
                                                        data=True))):
            if not conn['delay'] and neigh not in connected:
                connected.add(neigh)
                # Recurse through neighbours edges
                self.connected_without_delay(neigh, connected=connected)
        return connected

    def merge_nodes(self, nodes):
        """
        Merges a sub-graph of nodes into a single node represented by a
        multi-dynamics object. Used to merge nodes that are connected without
        delay, which necessitates simulaneous solution of their time
        derivatives

        Parameters
        ----------
        nodes : iterable(2-tuple(str, int))
            The indices of the nodes to merge into a single sub-component
        """
        sub_graph = self.graph.subgraph(nodes)
        # Create name for new combined multi-dynamics node from node
        # with the higest degree
        central_node = max(self.graph.degree(sub_graph.nodes),
                           key=itemgetter(1))[0]
        multi_node = (central_node[0] + '_multi', central_node[1])
        multi_name = sub_graph.nodes[central_node][
            'properties'].component_class.name + '_multi'
        self.graph.add_node(multi_node)
        # Group components with equivalent dynamics in order to assign
        # generic sub-component names based on sub-dynamics classes. This
        # should make the generated multi-dynamics class equalcla
        sub_components = defaultdict(list)
        for _, attr in sorted(sub_graph.nodes(data=True),
                              key=itemgetter(0)):
            # Add node to list of matching components
            component_class = attr['properties'].component_class
            matching = sub_components[component_class]
            matching.append(attr)
            # Get a uniuqe name for the sub-component based on its
            # dynamics class name + index
            attr['sub_comp'] = component_class.name
            if len(matching) > 1:
                attr['sub_comp'] += str(len(matching))
        # Map graph edges onto internal port connections of the new multi-
        # dynamics object
        port_connections = []
        edges_to_remove = []
        for u, v, conn in sub_graph.edges(data=True):
            if not conn['delay']:
                if conn['communicates'] == 'analog':
                    PortConnectionClass = AnalogPortConnection
                else:
                    PortConnectionClass = EventPortConnection
                port_connections.append(PortConnectionClass(
                    send_port_name=conn['src_port'],
                    receive_port_name=conn['dest_port'],
                    sender_name=sub_graph.nodes[u]['sub_comp'],
                    receiver_name=sub_graph.nodes[v]['sub_comp']))
                edges_to_remove.append((u, v))
        # Remove all edges in the sub-graph from the primary graph
        self.graph.remove_edges_from(edges_to_remove)
        # Redirect edges from merged multi-node to nodes external to the sub-
        # graph
        port_exposures = set()
        for u, v, conn in self.graph.out_edges(sub_graph, data=True):
            attr = self.graph.nodes[u]
            # Create a copy of conn dictionary so we can modify it
            conn = copy(conn)
            exposure = BasePortExposure.from_port(
                attr['properties'].component_class.port(conn['src_port']),
                attr['sub_comp'])
            port_exposures.add(exposure)
            conn['src_port'] = exposure.name
            self.graph.add_edge(multi_node, v, **conn)
        # Redirect edges to merged multi-node from nodes external to the sub-
        # graph
        for u, v, conn in self.graph.in_edges(sub_graph, data=True):
            attr = self.graph.nodes[v]
            # Create a copy of conn dictionary so we can modify it
            conn = copy(conn)
            exposure = BasePortExposure.from_port(
                attr['properties'].component_class.port(conn['dest_port']),
                attr['sub_comp'])
            port_exposures.add(exposure)
            conn['dest_port'] = exposure.name
            self.graph.add_edge(u, multi_node, **conn)
        # Create multi-dynamics object and set it as the properties object of
        # the new multi node
        multi_props = MultiDynamicsProperties(
            multi_name,
            sub_components={c['sub_comp']: c['properties']
                            for c in chain(*sub_components.values())},
            port_connections=port_connections,
            port_exposures=port_exposures,
            validate=False)
        # Remove merged nodes and their edges
        self.graph.remove_nodes_from(sub_graph)
        # Attempt to merge linear sub-components to limit the number of
        # states
        multi_props = multi_props.merge_states_of_linear_sub_components(
            validate=False)
        # Add merged node
        self.graph.nodes[multi_node]['properties'] = multi_props
