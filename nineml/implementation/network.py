from operator import itemgetter
from collections import defaultdict
from itertools import chain, repeat
from copy import copy
import numpy as np
import networkx as nx
import nineml.units as un
from pprint import pprint
from nineml.user import (
    MultiDynamicsProperties, AnalogPortConnection, EventPortConnection,
    BasePortExposure)
from .dynamics import Dynamics, DynamicsClass
from .utils import create_progress_bar


class Network(object):
    """
    An implementation of Network model in pure python

    Parameters
    ----------
    model : nineml.user.Network
        The model of the network described in 9ML
    start_t : Quantity(time)
        The starting time of the simulation
    """

    def __init__(self, model, start_t):
        component_arrays, connection_groups = model.flatten()
        # Initialise a graph to represent the network
        self.graph = nx.MultiDiGraph()
        # Add nodes (2-tuples consisting of <component-array-name> and
        # <cell-index>) for each component in each array
        for comp_array in component_arrays:
            props = comp_array.dynamics_properties
            for i in range(comp_array.size):
                self.graph.add_node((comp_array.name, i),
                                    properties=props.sample(i))
        # Add connections between components from connection groups
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
        # Merge dynamics definitions for nodes connected without delay
        # connections into multi-dynamics definitions. We save the iterator
        # into a list as we will be removing nodes as they are merged.
        for node in list(self.graph.nodes):
            if node not in self.graph:
                continue  # If node has already been merged
            conn_without_delay = self.sub_graph_connected_without_delay(node)
            if conn_without_delay:
                self.merge_nodes(conn_without_delay)
        # Initialise all dynamics components in graph
        dyn_class_cache = []  # Cache for storing previously analysed classes
        self.components = []
        for node, data in self.graph.nodes(data=True):
            # Attempt to reuse DynamicsClass objects between Dynamics objects
            # to save reanalysing their equations
            model = data['properties'].component_class
            try:
                dyn_class = next(dc for m, dc in dyn_class_cache if m == model)
            except StopIteration:
                dyn_class = DynamicsClass(model)
                dyn_class_cache.append((model, dyn_class))
            # Create dynamics object
            data['dynamics'] = dynamics = Dynamics(
                data['properties'], start_t, dynamics_class=dyn_class)
            self.components.append(dynamics)
        # Make all connections between dynamics components
        self.min_delay = float('inf')
        for u, v, conn in self.graph.out_edges(data=True):
            from_dyn = self.graph.nodes[u]['dynamics']
            to_dyn = self.graph.nodes[v]['dynamics']
            delay = conn['delay']
            from_dyn.ports[conn['src_port']].connect_to(
                to_dyn.analog_receive_ports[conn['dest_port']],
                delay=delay)
            if delay < self.min_delay:
                self.min_delay = delay

    def simulate(self, stop_t, dt, progress_bar=True):
        stop_t = float(stop_t.in_units(un.s))
        dt = float(dt.in_si_units())
        if progress_bar is True:
            progress_bar = create_progress_bar(self.t, stop_t, self.min_delay)
        slice_dt = min(stop_t, self.min_delay)
        for t in np.arange(self.t, stop_t, slice_dt):
            for component in self.components:
                component.simulate(t, dt)
            if progress_bar is not None:
                progress_bar.update(t)
        if progress_bar is not None:
            progress_bar.finish()

    def sub_graph_connected_without_delay(self, start_node, connected=None):
        """
        Returns the sub-graph of nodes connected to the current node by
        (potentially multiple) delayless connections

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
                self._get_connected_without_delay(neigh, connected=connected)
        return connected

    def merge_nodes(self, nodes):
        """
        Merges a sub-graph of nodes into a single node represented by a
        multi-dynamics object. Used to merge nodes that are connected without
        delay and therefore there equations need to be solved simultaneously

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
        for _, data in sorted(sub_graph.nodes(data=True),
                              key=itemgetter(0)):
            # Add node to list of matching components
            component_class = data['properties'].component_class
            matching = sub_components[component_class]
            matching.append(data)
            # Get a uniuqe name for the sub-component based on its
            # dynamics class name + index
            data['sub_comp'] = component_class.name
            if len(matching) > 1:
                data['sub_comp'] += str(len(matching))
        # Map graph edges onto internal port connections of the new multi-
        # dynamics object
        port_connections = []
        for u, v, conn in sub_graph.edges(data=True):
            if conn['communicates'] == 'analog':
                PortConnectionClass = AnalogPortConnection
            else:
                PortConnectionClass = EventPortConnection
            port_connections.append(PortConnectionClass(
                send_port_name=conn['src_port'],
                receive_port_name=conn['dest_port'],
                sender_name=sub_graph.nodes[u]['sub_comp'],
                receiver_name=sub_graph.nodes[v]['sub_comp']))
        # Remove all edges in the sub-graph from the primary graph
        self.graph.remove_edges_from(list(sub_graph.edges))
        # Redirect edges from merged multi-node to nodes external to the sub-
        # graph
        port_exposures = set()
        for u, v, conn in self.graph.out_edges(sub_graph, data=True):
            data = self.graph.nodes[u]
            # Create a copy of conn dictionary so we can modify it
            conn = copy(conn)
            exposure = BasePortExposure.from_port(
                data['properties'].component_class.port(conn['src_port']),
                data['sub_comp'])
            port_exposures.add(exposure)
            conn['src_port'] = exposure.name
            self.graph.add_edge(multi_node, v, **conn)
        # Redirect edges to merged multi-node from nodes external to the sub-
        # graph
        for u, v, conn in self.graph.in_edges(sub_graph, data=True):
            data = self.graph.nodes[v]
            # Create a copy of conn dictionary so we can modify it
            conn = copy(conn)
            exposure = BasePortExposure.from_port(
                data['properties'].component_class.port(conn['dest_port']),
                data['sub_comp'])
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
            validate=True)
        # Add merged node
        self.graph.nodes[multi_node]['properties'] = multi_props
