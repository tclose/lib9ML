from itertools import repeat
import networkx as nx


class Network(object):

    def __init__(self, model, initial_states, start_t, dt):
        expanded_graph = nx.Graph()
        # Add nodes (2-tuples consisting of <component-array-name> and
        # <cell-index>) for each component in each array
        for comp_array in model.component_arrays:
            expanded_graph.add_nodes_from(zip(repeat(comp_array.name),
                                          range(len(comp_array))))
        # Add connections between components from connection groups
        for conn_group in model.connection_groups:
            expanded_graph.add_edges_from(
                ((conn_group.source, c[0]),
                 (conn_group.destination, c[1]),
                 {'delay': d})
                for c, d in zip(conn_group.connections, conn_group.delay))

    def simulate(self, stop_t):
        pass
