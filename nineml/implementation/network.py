from operator import itemgetter
from collections import defaultdict
from itertools import chain
import numpy as np
import networkx as nx
import nineml.units as un
from nineml.user import (
    MultiDynamicsProperties, AnalogPortConnection, EventPortConnection)
from .dynamics import Dynamics, DynamicsClass
from .utils import create_progress_bar


class Network(object):

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
            self.graph.add_edges_from(
                ((conn_group.source, c[0]),
                 (conn_group.destination, c[1]),
                 {'communicates': conn_group.communicates,
                  'delay': d,
                  'source_port': conn_group.source_port,
                  'destination_port': conn_group.destination_port})
                for c, d in zip(conn_group.connections, conn_group.delay))
        # Merge dynamics definitions for nodes connected with zero delay
        # connections into multi-dynamics definitions
        self._merge_nodes_connected_with_no_delay()
        # Initialise all dynamics components in graph
        dyn_class_cache = {}  # Cache for storing previously analysed classes
        self.components = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            # Attempt to reuse DynamicsClass objects between Dynamics objects
            # to save reanalysing their equations
            cc = node_data['properties'].component_class
            try:
                dyn_class = dyn_class_cache[cc]
            except KeyError:
                dyn_class = DynamicsClass(cc)
                dyn_class_cache[cc] = dyn_class
            # Create dynamics object
            node_data['dynamics'] = dynamics = Dynamics(
                dyn_class,
                node_data['properties'], start_t)
            self.components.append(dynamics)
        # Make all connections between dynamics components
        self.min_delay = float('inf')
        for node in self.graph.nodes():
            from_dyn = self.graph.nodes[node]['dynamics']
            for successor in self.graph.successors(node):
                to_dyn = self.graph.nodes[successor]['dynamics']
                for conn_data in self.graph.get_edge_data(successor):
                    delay = conn_data['delay']
                    from_dyn.ports[conn_data['source_port']].connect_to(
                        to_dyn.analog_receive_ports[
                            conn_data['destination_port']],
                        delay=delay)
                    if delay < self.min_delay:
                        self.min_delay = delay

    def _merge_nodes_connected_with_no_delay(self):
        for node in list(self.graph.nodes()):
            if node not in self.graph:
                continue  # If it has already been merged
            to_merge = self._neighbours_to_merge(node, set())
            if to_merge:
                to_merge.add(node)
                sub_graph = self.graph.subgraph(to_merge)
                # Get node with highest degree in order to name the new
                # merge multi-dynamics object
                centre_node = max(sub_graph.degree(), key=itemgetter(1))[0]
                # Get dictionary of all sub-components to merge
                sub_comp_names = {}
                sub_comp_props = {}
                counters = defaultdict(lambda: 0)  # For gen. unique names
                for sub_comp_node in to_merge:
                    node_data = sub_graph[sub_comp_node]
                    # Get a uniuqe name for the sub-component based on the
                    # component array it comes from
                    comp_array_name = sub_comp_node[0]
                    sub_comp_name = '{}{}'.format(
                        comp_array_name, counters[comp_array_name])
                    counters[comp_array_name] += 1
                    sub_comp_names[sub_comp_node] = sub_comp_name
                    # Add properties to list of components
                    sub_comp_props[sub_comp_name] = node_data['properties']
                # Map graph edges to internal port connections within
                # the new multi-dynamics object
                port_connections = []
                for conn in sub_graph.edges():
                    conn_data = sub_graph.get_edge_data(conn)
                    if conn_data['communicates'] == 'analog':
                        PortConnectionClass = AnalogPortConnection
                    else:
                        PortConnectionClass = EventPortConnection
                    port_connections.append(PortConnectionClass(
                        send_port_name=conn_data['source_port'],
                        receive_port_name=conn_data['destination_port'],
                        sender_name='{}{}'.format(*conn[0]),
                        receiver_name='{}{}'.format(*conn[1])))
                multi_name = (sub_graph[centre_node]['properties'].name +
                              '__multi')
                multi_props = MultiDynamicsProperties(
                    multi_name,
                    sub_components=sub_comp_props,
                    port_connections=port_connections)
                # Add new combined multi-dynamics node
                multi_node = (centre_node[0] + '_multi', centre_node[1])
                self._graph.add_node(
                    multi_node,
                    properties=multi_props)
                # Redirect edges into/from merged nodes to multi-node
                for sub_comp_node in to_merge:
                    for pcessor in self.graph.predecessors(sub_comp_node):
                        if pcessor not in to_merge:
                            for conn_data in self.graph.get_edge_data(
                                    pcessor, sub_comp_node):
                                conn_data['destination_port'] = (
                                    '{}__{}'.format(
                                        conn_data['destination_port'],
                                        sub_comp_names[sub_comp_node]))
                                self.graph.add_edge(
                                    pcessor,
                                    multi_node,
                                    conn_data)
                    for scessor in self.graph.successors(sub_comp_node):
                        if scessor not in to_merge:
                            for conn_data in self.graph.get_edge_data(
                                    sub_comp_node, scessor):
                                conn_data['source_port'] = '{}__{}'.format(
                                    conn_data['source_port'],
                                    sub_comp_names[sub_comp_node])
                                self.graph.add_edge(
                                    pcessor,
                                    multi_node,
                                    conn_data)
                # Remove merged nodes
                self.graph.remove_nodes_from(to_merge)

    def _neighbours_to_merge(self, node, to_merge):
        """
        Recursively adds all nodes connected to a given node (and nodes
        connected to them) by zero delay
        """
        for neighbour in chain(self.graph.predecessors(node),
                               self.graph.successors(node)):
            if neighbour not in to_merge:
                for conn_data in self.graph.get_edge_data(node, neighbour):
                    if conn_data['delay'] == 0.0 * un.ms:
                        to_merge.add(neighbour)
                        self._neighbours_to_merge(neighbour, to_merge)

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
