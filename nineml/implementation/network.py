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
from nineml.abstraction.dynamics.visitors.queriers import DynamicsAreLinear
from .dynamics import Dynamics, DynamicsClass
from .utils import create_progress_bar


class Network(object):

    def __init__(self, model, start_t):
        component_arrays, connection_groups = model.flatten(
            combine_cell_with_synapses=True, merge_linear_synapses=True)
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

            without_delay = set()

            def connected_without_delay(n):
                # Iterate all in-coming and out-going edges and check for
                # any zero delays
                for neigh, conns in chain(
                    (n, c) for n, _, c in self.graph.in_edges(n, data=True),
                        (n, c) for _, n, c in self.graph.out_edges(n,
                                                                   data=True)):
                    if neigh not in without_delay:
                        if any(c['delay'] for c in conns.values()):
                            without_delay.add(neigh)
                            # Recurse through neighbours edges
                            connected_without_delay(neigh)

            if without_delay:
                self._merge_sub_graph(self.graph.subgraph(without_delay))
        # Initialise all dynamics components in graph
        dyn_class_cache = {}  # Cache for storing previously analysed classes
        self.components = []
        for node, data in self.graph.nodes(data=True):
            # Attempt to reuse DynamicsClass objects between Dynamics objects
            # to save reanalysing their equations
            cc = data['properties'].dynamics_class
            try:
                dyn_class = dyn_class_cache[cc]
            except KeyError:
                dyn_class_cache[cc] = dyn_class = DynamicsClass(cc)
            # Create dynamics object
            data['dynamics'] = dynamics = Dynamics(
                data['properties'], start_t, array_index=data['array_index'],
                dynamics_class=dyn_class)
            self.components.append(dynamics)
        # Make all connections between dynamics components
        self.min_delay = float('inf')
        for u, v, conns in self.graph.out_edges(data=True):
            from_dyn = self.graph.nodes[u]['dynamics']
            to_dyn = self.graph.nodes[v]['dynamics']
            for conn in conns.values():
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

    def _merge_sub_graph(self, sub_graph, combine_linear=False):
        """
        Merges a sub-graph of nodes into a single node represented by a
        multi-dynamics object. Used to merge nodes that are connected without
        delay and therefore there equations need to be solved simultaneously

        Parameters
        ----------
        sub_graph : nx.MultiDiGraph
            A sub-graph of self.graph that is to be merged
        combine_linear : bool
            Flag whether to combine linear dynamics into single component
        """
        # Create name for new combined multi-dynamics node from node
        # with the higest degree
        central_node = max(self.graph.degree(sub_graph.nodes),
                           key=itemgetter(1))
        multi_node = (central_node[0] + '_multi', central_node[1])
        self.graph.add_node(multi_node)
        # Group components with equivalent dynamics in order to assign
        # generic sub-component names based on sub-dynamics classes
        sub_components = defaultdict(list)
        for sc_node, data in sub_graph.nodes(data=True):
            # Add node to list of matching components
            component_class = data['properties'].component_class
            matching = sub_components[component_class]
            matching.append(sc_node)
            # Get a uniuqe name for the sub-component based on its
            # dynamics class name + index
            data['sub_comp'] = component_class.name
            if len(matching) > 1:
                data['sub_comp'] += str(len(matching))
        # Map graph edges onto internal port connections of the new multi-
        # dynamics object
        port_connections = []
        for u, v, conns in sub_graph.edges(data=True):
            for conn in conns.values():
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
        self.graph.remove_edges_from(sub_graph.edges)
        # Redirect edges from merged multi-node to nodes external to the sub-
        # graph
        port_exposures = set()
        for u, v, conns in self.graph.out_edges(sub_graph, data=True):
            data = self.graph.nodes[u]
            for conn in conns.values():
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
        for u, v, conns in self.graph.in_edges(sub_graph, data=True):
            data = self.graph.nodes[v]
            for conn in conns.values():
                # Create a copy of conn dictionary so we can modify it
                conn = copy(conn)
                exposure = BasePortExposure.from_port(
                    data['properties'].component_class.port(conn['dest_port']),
                    data['sub_comp'])
                port_exposures.add(exposure)
                conn['dest_port'] = exposure.name
                self.graph.add_edge(u, multi_node, **conn)
        # Remove merged nodes and their edges
        self.graph.remove_nodes_from(sub_graph)
        # Attempt to combine sub-components with equivalent linear dynamics in
        # order to reduce the number of state-variables in the multi-dynamics
        if combine_linear:
            for comp_class, nodes in sub_components.items():
                query = DynamicsAreLinear(comp_class)
                matching_td_props = defaultdict(list)
                if query.linear:
                    # Sort components into matching properties used in ODES
                    for node in nodes:
                        props = node['properties']
                        td_props = frozenset(
                            props[p] for p in query.time_derivative_parameters)
                        matching_td_props[td_props].append(props)
                for td_props, matches in matching_td_props.items():
                    if len(matches) > 1:
                        pass
        # Create multi-dynamics object and set it as the properties object of
        # the new multi node
        multi_dyn_props = MultiDynamicsProperties(
            sub_graph.nodes[central_node]['combponent_class'].name + '_multi',
            sub_components={n['sub_comp']: n['properties']
                            for n in chain(sub_components.values())},
            port_connections=port_connections,
            port_exposures=port_exposures)
        self.graph.nodes[multi_node]['properties'] = multi_dyn_props
