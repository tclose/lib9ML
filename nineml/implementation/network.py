import networkx as nx
import nineml.units as un


class Network(object):

    def __init__(self, model, initial_states, start_t, dt):
        self.graph = nx.DiGraph()
        # Add nodes (2-tuples consisting of <component-array-name> and
        # <cell-index>) for each component in each array
        for comp_array in model.component_arrays:
            props = comp_array.dynamics_properties
            for i in range(comp_array.size):
                self.graph.add_node(
                    (comp_array.name, i),
                    properties=props.single(i),
                    initial_state=initial_states[comp_array.name][i])
        # Add connections between components from connection groups
        for conn_group in model.connection_groups:
            self.graph.add_edges_from(
                ((conn_group.source, c[0]),
                 (conn_group.destination, c[1]),
                 {'conn_group': conn_group.name,
                  'delay': d,
                  'source_port': conn_group.source_port,
                  'destination_port': conn_group.destination_port})
                for c, d in zip(conn_group.connections, conn_group.delay))
        # Merge nodes with zero delay connections into multi-dynamics nodes
        for node in list(self.graph.nodes()):
            nodes_to_merge = []
            for neigh_node in self.graph.neighbors(node):
                conn_data = self.graph.get_edge_data(node, neigh_node)
                if conn_data['delay'] == 0.0 * un.ms:
                    nodes_to_merge.append(neigh_node)

    def simulate(self, stop_t):
        pass
