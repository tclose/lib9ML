from future.utils import PY2
from operator import attrgetter
from collections import defaultdict, namedtuple, Counter
from itertools import chain, repeat
from logging import getLogger
import multiprocessing as mp
import nineml.units as un
from nineml.user import (
    MultiDynamicsProperties, AnalogPortConnection, EventPortConnection,
    BasePortExposure)
from nineml.units import Quantity
from .dynamics import (
    Dynamics, DynamicsClass, AnalogSendPort, EventSendPort)
from .experiment import AnalogSource, EventSource, AnalogSink, EventSink
from tqdm import tqdm
from nineml.abstraction.dynamics.visitors.modifiers import (
    DynamicsMergeStatesOfLinearSubComponents)
from nineml.exceptions import (
    NineMLUsageError, NineMLCannotMergeException, NineMLInternalError)
from pprint import pprint, pformat


logger = getLogger('nineml')


graph_size = 0
prev_incr = 0

debug_merge_mem = True


class Network(object):
    """
    An implementation of Network model in pure Python. All populations and
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
    show_progress : bool
        Whether to show progress of network construction
    num_processes : int
        The number of processes to create the network over. If != 1 then
        initialisation of the network is delayed until the first (and only
        allowable) call to simulate.
    """

    def __init__(self, model, start_t, sources=None, sinks=None,
                 show_progress=True, num_processes=1):
        if isinstance(start_t, Quantity):
            start_t = float(start_t.in_si_units())
        if sources is None:
            sources = []
        if sinks is None:
            sinks = []
        if num_processes is None:
            self.num_procs = mp.cpu_count()
        elif num_processes < 1:
            raise NineMLUsageError(
                "Cannot not specify less than one processor to use ({})"
                .format(num_processes))
        else:
            self.num_procs = num_processes
        self.t = start_t
        self.model = model
        component_arrays, connection_groups = model.flatten()
        # Initialise a graph to represent the network
        progress_bar = tqdm(
            total=sum(ca.size for ca in component_arrays),
            desc="Adding nodes to network graph",
            disable=not show_progress)
        graph = NetworkGraph()
        # Add nodes (2-tuples consisting of <component-array-name> and
        # <cell-index>) for each component in each array
        ca_dict = {}
        for comp_array in component_arrays:
            ca_dict[comp_array.name] = comp_array
            props = comp_array.dynamics_properties
            for i in range(comp_array.size):
                graph.add_node(comp_array.name, i, i, props,
                               (i % self.num_procs))
                progress_bar.update()
        progress_bar.close()
        # Add connections between components from connection groups
        self.min_delay = float('inf')
        progress_bar = tqdm(
            total=sum(len(cg) for cg in connection_groups),
            desc="Adding edges to network graph",
            disable=not show_progress)
        for conn_group in connection_groups:
            delay_qty = conn_group.delay
            if delay_qty is None:
                delays = repeat(0.0)
            else:
                delays = (float(d) for d in delay_qty.in_si_units())
            for (src_i, dest_i), delay in zip(conn_group.connections, delays):
                graph.add_edge(
                    graph.node(conn_group.source.name, int(src_i)),
                    graph.node(conn_group.destination.name, int(dest_i)),
                    src_port=conn_group.source_port,
                    dest_port=conn_group.destination_port,
                    communicates=conn_group.communicates,
                    delay=delay)
                if delay and delay < self.min_delay:
                    self.min_delay = delay
                progress_bar.update()
        progress_bar.close()
        # Add sources to network graph
        self.sources = defaultdict(list)
        for comp_array_name, port_name, index, signal in sources:
            port = ca_dict[comp_array_name].dynamics_properties.port[port_name]
            graph.add_source(
                comp_array_name, index, signal,
                communicates=port.communicates,
                dest_port=port_name)
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
            comp_array = ca_dict[comp_array_name]
            port = comp_array.dynamics_properties.port(port_name)
            for index in indices:
                if index >= comp_array.size:
                    continue  # Skip this sink as it is out of bounds
                graph.add_sink(
                    comp_array_name, index,
                    src_port=port_name,
                    communicates=port.communicates,
                    delay=self.min_delay)
        # Merge dynamics definitions for nodes connected without delay
        # connections into multi-dynamics definitions. We save the iterator
        # into a list as we will be removing nodes as they are merged.
        progress_bar = tqdm(
            total=len(graph),
            desc="Merging sub-graphs with delayless connections",
            disable=not show_progress)
        self.cached_merged = []
        self.cached_mergers = []
        for node in list(graph.nodes):
            if node not in graph:
                continue  # If node has already been merged
            conn_without_delay = self.connected_without_delay(node)
            num_to_merge = len(conn_without_delay)
            if num_to_merge > 1:
                graph = self.merge_nodes(conn_without_delay, graph)
            progress_bar.update(num_to_merge)
        progress_bar.close()
        if self.num_procs == 1:
            # Initialise all dynamics components in graph
            (self.components, self.sources,
             self.sinks) = self._initialise_components(graph, start_t,
                                                       show_progress)
        else:
            # Split network over number of specified processes

            # Construction of components is delayed until simulation step
            # so they can be constructed in child processes
            self.components = None
            # Construct inter-process pipes between each pair of processes and
            # wrap in remote receiver/sender objects to handle inter-process
            # communication
            self.remote_senders = {i: {} for i in range(self.num_procs)}
            self.remote_receivers = {i: {} for i in range(self.num_procs)}
            for i in range(self.num_procs):
                for j in range(self.num_procs):
                    if i != j:
                        send_end, receive_end = mp.Pipe()
                        self.remote_senders[i][j] = RemoteSender(
                            i, j, send_end)
                        self.remote_receivers[j][i] = RemoteReceiver(
                            i, j, receive_end)
            # Replace edges that span processes with remote send/receivers
            edges_to_remove = []
            for edge in graph.edges:
                src_node = edge.src_node
                dest_node = edge.dest_node
                if src_node.rank != dest_node.rank:
                    # Record port to send from and remote receivers object to
                    # send it to when the node is constructed as well as the
                    # max delay (for determining the length of the required
                    # buffer
                    key = (edge.src_port, self.remote_senders[
                        src_node.rank][dest_node.rank])
                    try:
                        delay = src_node.remote_receive_ports[key]
                    except KeyError:
                        delay = -1
                    if edge.delay >= delay:
                        src_node.remote_receive_ports[key] = edge.delay
                    # Record port to receive to, the remote senders object to
                    # receive it from and the key of the sending component/port
                    dest_node.remote_send_ports.append(
                        ((src_node[:2], edge.src_port),
                         edge.dest_port,
                         delay,
                         self.remote_receivers[src_node.rank][dest_node.rank]))
            graph.remove_edges(edges_to_remove)
            # Divide up network graph between available processes
            self.sub_graphs = []
            for rank in range(self.num_procs):
                # Copy nodes in the sub to a new graph to avoid referencing the
                # original graph when it is pickled and sent to a new process
                sub_graph = NetworkGraph(n for n in graph.nodes
                                         if n.rank == rank)
                self.sub_graphs.append(sub_graph)
                # Delete sub_graph nodes from original graph to free up memory
                graph.remove_nodes(sub_graph.nodes)

    @property
    def name(self):
        return self.model.name

    def simulate(self, stop_t, dt, show_progress=True):
        """
        Simulate the network until stop_t

        Parameters
        ----------
        stop_t : Quantity (time) | float (s)
            The time to simulate the network until
        dt : Quantity (time) | float (s)
            The timestep for the components
        show_progress : bool
            Display a progress bar.
        """
        if isinstance(stop_t, Quantity):
            stop_t = float(stop_t.in_units(un.s))
        if isinstance(dt, Quantity):
            dt = float(dt.in_si_units())
        if self.num_procs == 1:
            # Simulate all components on the same node
            self._simulate(self.components, self.t, stop_t, dt, self.min_delay,
                           self.model.name, show_progress)
        else:
            # Create child processes to simulate a sub-graph of the network
            processes = [
                mp.Process(
                    target=self._simulate_sub_graph,
                    args=(sg, self.t, stop_t, dt, self.min_delay,
                          self.model.name, False, self.remote_senders[i],
                          self.remote_receivers[i], i))
                for i, sg in enumerate(self.sub_graphs[1:], start=1)]
            # Start all child processes
            for p in processes:
                p.start()
            # Simulate the first sub-graph on the master process
            self.sinks = self._simulate_sub_graph(
                self.sub_graphs[0], self.t, stop_t, dt, self.min_delay,
                self.model.name, show_progress, self.remote_senders[0],
                self.remote_receivers[0], 0)
            # Join all child processes
            for p in processes:
                p.join()
        self.t = stop_t
        return self.sinks

    @classmethod
    def _simulate(cls, components, t, stop_t, dt, min_delay, model_name,
                  show_progress, remote_senders=None,
                  remote_receivers=None, rank=None):
        # Get the schedule for communicating with other processes
        if rank is not None:
            comm_schedule = cls.interprocess_comm_schedule(
                len(remote_receivers) + 1)[rank]
        progress_bar = tqdm(
            initial=t, total=stop_t,
            desc=("Simulating '{}' network (dt={} s)".format(model_name, dt)),
            unit='s (sim)', unit_scale=True,
            disable=not show_progress)
        while t < stop_t:
            new_t = min(stop_t, t + min_delay)
            slice_dt = new_t - t
            for component in components:
                component.simulate(new_t, dt, show_progress=False)
            # Perform inter-process communication if required
            if rank is not None:
                for pairing in comm_schedule:
                    if None in pairing:
                        continue   # "bye"
                    if rank == pairing[0]:
                        # If "home" team, send data first
                        remote = pairing[1]
                        logger.debug('Sending data from {} to {}'
                                     .format(rank, remote))
                        remote_senders[remote].send_data()
                        logger.debug('Receiving data from {} to {}'
                                     .format(remote, rank))
                        remote_receivers[remote].receive_data()
                    else:
                        # If "away" team, receive data first
                        remote = pairing[0]
                        logger.debug('Receiving data from {} to {}'
                                     .format(remote, rank))
                        remote_receivers[remote].receive_data()
                        logger.debug('Sending data from {} to {}'
                                     .format(rank, remote))
                        remote_senders[remote].send_data()
            progress_bar.update(slice_dt)
            t = new_t
        progress_bar.close()

    @classmethod
    def _simulate_sub_graph(cls, sub_graph, t, stop_t, dt, min_delay,
                            model_name, show_progress, remote_senders,
                            remote_receivers, rank):
        components, sinks, _ = cls._initialise_components(
            sub_graph, t, show_progress)
        cls._simulate(components, t, stop_t, dt, min_delay, model_name,
                      show_progress, remote_senders, remote_receivers, rank)
        return sinks

    @classmethod
    def _initialise_components(cls, graph, start_t, show_progress):
        components = []
        sources = defaultdict(list)
        sinks = defaultdict(list)
        dyn_class_cache = []  # Cache for storing previously analysed classes
        for node in tqdm(graph.nodes, desc="Iniitalising dynamics",
                         disable=not show_progress):
            if node.type == 'node':
                model = node.props.component_class
                try:
                    # Attempt to reuse DynamicsClass objects between Dynamics
                    # objects to save reanalysing their equations
                    dyn_class = next(dc for m, dc in dyn_class_cache
                                     if m == model)
                except StopIteration:
                    dyn_class = DynamicsClass(model)
                    dyn_class_cache.append((model, dyn_class))
                # Create dynamics object
                component = Dynamics(
                    node.props, start_t, dynamics_class=dyn_class,
                    name='{}_{}'.format(*node.key),
                    sample_index=node.sample_index,
                    rank=node.rank)
                components.append(component)
            elif node.type == 'source':
                if node.edge.communicates == 'analog':
                    cls = AnalogSource
                else:
                    cls = EventSource
                connected_to = node.connected_to
                source_name = '{}_{}_{}' .format(
                    connected_to.comp_array, node.edge.dest_port,
                    connected_to.index)
                component = cls(source_name, node.signal)
                sources[connected_to.comp_array].append(component)
            elif node.type == 'sink':
                if node.edge.communicates == 'analog':
                    cls = AnalogSink
                else:
                    cls = EventSink
                connected_to = node.connected_to
                sink_name = '{}_{}_{}' .format(
                    connected_to.comp_array, node.edge.src_port,
                    connected_to.index)
                component = cls(sink_name)
                sinks[connected_to.comp_array].append(component)
            else:
                assert False, "Unrecognised node type '{}'".format(node.type)
            node.component = component
            # Make all remote connections from send ports (if applicable)
            for ((port_name, remote_sender),
                 max_delay) in node.remote_receive_ports.items():
                remote_sender.connect(node, node.component.port(port_name),
                                      max_delay)
            # Make all remote connections to receive ports (if applicable)
            for (remote_key, port_name,
                 delay, remote_receiver) in node.remote_send_ports:
                remote_receiver.connect(remote_key,
                                        node.component.port(port_name), delay)
        # Make all connections between dynamics components, sources and sinks
        for edge in tqdm(graph.edges, desc="Connecting components",
                         disable=not show_progress):
            from_port = edge.src_node.component.port(edge.src_port)
            to_port = edge.dest_node.component.port(edge.dest_port)
            from_port.connect_to(to_port, delay=edge.delay)
        return components, sources, sinks

    def connected_without_delay(self, start_node):
        """
        Returns the sub-graph of nodes connected to the start node by a chain
        of delayless connections.

        Parameters
        ----------
        start_node : tuple(str, int)
            The starting node to check for delayless connected neighbours from
        graph : nx.MultiDiGraph
            The network graph
        connected : set(tuple(str, int))
            The set eventually returned by the method, passed as a arg to
            recursive calls of this method
        """
        connected = set([start_node])
        stack = [start_node]
        # Iterate all in-coming and out-going edges and check for
        # any zero delays. If so, add to set of nodes to merge
        while stack:
            node = stack.pop()
            for neighbour, edge in chain(((e.src_node, e)
                                          for e in node.in_edges),
                                         ((e.dest_node, e)
                                          for e in node.out_edges)):
                if not edge.delay and neighbour not in connected:
                    connected.add(neighbour)
                    stack.append(neighbour)
        return connected

    def merge_nodes(self, nodes_to_merge, graph):
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
        # Create name for new combined multi-dynamics node from node
        # with the higest degree
        central_node = None
        max_degree = -1
        for node in nodes_to_merge:
            degree = (len([e for e in node.out_edges
                           if e.dest_node in nodes_to_merge]) +
                      len([e for e in node.in_edges
                           if e.src_node in nodes_to_merge]))
            if degree > max_degree:
                central_node = node
                max_degree = degree
        multi_name = central_node.props.component_class.name + '_multi'
        # Group components with equivalent dynamics in order to assign
        # generic sub-component names based on sub-dynamics classes. This
        # should make the generated multi-dynamics class equalcla
        sub_comp_counters = Counter()
        sample_indices = {}
        for node in sorted(nodes_to_merge, key=attrgetter('key')):
            # Add node to list of matching components
            component_class = node.props.component_class
            sub_comp_counters.update([component_class])
            count = sub_comp_counters[component_class]
            # Get a uniuqe name for the sub-component based on its
            # dynamics class name + index
            node.sub_comp_name = component_class.name
            if count > 1:
                node.sub_comp_name += str(count)
            sample_indices[node.sub_comp_name] = node.sample_index
        # Add node but delay setting its props until after they are merged
        multi_node = graph.add_node(
            central_node.comp_array + '_multi', central_node.index,
            sample_indices, None, central_node.rank)
        # Map graph edges onto internal port connections of the new multi-
        # dynamics object
        port_connections = []
        port_exposures = set()
        # Redirect edges from merged multi-node to nodes external to the sub-
        # graph
        for node in nodes_to_merge:
            for edge in node.out_edges:
                if edge.dest_node in nodes_to_merge:
                    if edge.communicates == 'analog':
                        PortConnectionClass = AnalogPortConnection
                    else:
                        PortConnectionClass = EventPortConnection
                    port_connections.append(PortConnectionClass(
                        send_port_name=edge.src_port,
                        receive_port_name=edge.dest_port,
                        sender_name=edge.src_node.sub_comp_name,
                        receiver_name=edge.dest_node.sub_comp_name))
                else:
                    # Create a copy of conn dictionary so we can modify it
                    exposure = BasePortExposure.from_port(
                        node.props.component_class.port(edge.src_port),
                        node.sub_comp_name)
                    port_exposures.add(exposure)
                    graph.add_edge(multi_node, edge.dest_node,
                                   exposure.name, edge.dest_port,
                                   edge.communicates, edge.delay)
            for edge in node.in_edges:
                # If incoming edge to sub-graph
                if edge.src_node not in nodes_to_merge:
                    exposure = BasePortExposure.from_port(
                        node.props.component_class.port(edge.dest_port),
                        node.sub_comp_name)
                    port_exposures.add(exposure)
                    graph.add_edge(edge.src_node, multi_node,
                                   edge.src_port, exposure.name,
                                   edge.communicates, edge.delay)
        # Create multi-dynamics object and set it as the properties object of
        # the new multi node
        multi_props = MultiDynamicsProperties(
            multi_name,
            sub_components={n.sub_comp_name: n.props for n in nodes_to_merge},
            port_connections=port_connections,
            port_exposures=port_exposures,
            validate=False)
        # Remove merged nodes and their edges
        graph.remove_nodes(nodes_to_merge)
        # Attempt to merge linear sub-components to limit the number of
        # states
        merged = None
        for prev_props, cached in self.cached_merged:
            if prev_props == multi_props:
                merged = cached
                break
        if merged is None:
            for cached_merger in self.cached_mergers:
                try:
                    merged = cached_merger.merge(multi_props)
                except NineMLCannotMergeException:
                    continue
                else:
                    break
        if merged is None:
            merger = DynamicsMergeStatesOfLinearSubComponents(
                multi_props, validate=False)
            merged = merger.merged
            self.cached_mergers.append(merger)
        self.cached_merged.append((multi_props, merged))
        del multi_props
        # Add merged node
        # Set multi props
        multi_node.props = merged
        return graph

    @classmethod
    def interprocess_comm_schedule(cls, num_procs):
        """
        Return schedule for inter-process communication based on a round-robin
        tournament scheduling algorithm
        """
        procs = list(range(num_procs))

        # To make even number of pairings introduce a "bye"
        if num_procs % 2:
            procs.append(None)
        num_procs = len(procs)

        schedule = {i: [] for i in procs}
        for _ in range(num_procs - 1):
            for i in range(num_procs // 2):
                pair = (procs[i], procs[- i - 1])
                schedule[pair[0]].append(pair)
                schedule[pair[1]].append(pair)
            procs.insert(1, procs.pop())

        return schedule


class RemoteCommunicator(object):

    def __init__(self, send_rank, receive_rank, pipe_end):
        self.send_rank = send_rank
        self.receive_rank = receive_rank
        self.pipe_end = pipe_end
        self.ports = {}

    def __hash__(self):
        return hash((self.end, self.send_rank, self.receive_rank))

    def __eq__(self, other):
        return (self.end == other.end and
                self.send_rank == other.send_rank and
                self.receive_rank == other.receive_rank)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "{}(send={}, receive={})".format(type(self).__name__,
                                                self.send_rank,
                                                self.receive_rank)


class RemoteSender(RemoteCommunicator):

    end = 'receive'

    def send_data(self):
        logger.debug("Sending data from {} to {}'".format(self.send_rank,
                                                          self.receive_rank))
        self.pipe_end.send([(k, p.data) for k, p in self.ports.items()
                            if p.data])

    def connect(self, network_id, port, delay):
        key = (network_id, port.name)
        assert key not in self.ports
        if port.communicates == 'analog':
            remote_port = RemoteAnalogReceivePort(key)
        else:
            remote_port = RemoteEventReceivePort(key)
        self.ports[key] = remote_port
        port.connect_to(remote_port, delay)


class RemoteReceiver(RemoteCommunicator):

    end = 'send'

    def receive_data(self):
        logger.debug("Receiving data at {} from {}'".format(self.receive_rank,
                                                            self.send_rank))
        for key, data in self.pipe_end.recv():
            try:
                self.ports[key].update(data)
            except KeyError:
                raise NineMLInternalError(
                    "Did not find port corresponding to {} key:\n{}"
                    .format(key, pformat(sorted(self.ports.keys()))))

    def connect(self, remote_key, port, delay):
        try:
            remote_port = self.ports[remote_key]
        except KeyError:
            if port.communicates == 'analog':
                remote_port = RemoteAnalogSendPort(remote_key)
            else:
                remote_port = RemoteEventSendPort(remote_key)
            self.ports[remote_key] = remote_port
        else:
            assert port.communicates == remote_port.communicates
        remote_port.connect_to(port, delay)
        return port


class RemoteEventReceivePort(object):

    communicates = 'event'

    def __init__(self, name):
        self._name = name
        self.events = []

    def receive(self, t):
        self.events.append(t)  # We assume that the events will come in order

    @property
    def data(self):
        return self.events

    def update_buffer(self):
        pass

    @property
    def name(self):
        return self._name


class RemoteAnalogReceivePort(object):

    communicates = 'analog'

    def __init__(self, name):
        self._name = name
        self.send_port = None

    def connect_from(self, send_port, delay=None):  # @UnusedVariable
        # Note delay is not used on the local side of the connection
        self.send_port = send_port

    @property
    def data(self):
        return self.send_port.buffer

    def update_buffer(self):
        pass

    @property
    def name(self):
        return self._name


class RemoteAnalogSendPort(AnalogSendPort):

    def __init__(self, name):
        self._name = name
        self.receivers = []
        self.buffer = None
        self.max_delay = 0.0

    def update(self, buffer):
        self.buffer = buffer

    @property
    def name(self):
        return self._name


class RemoteEventSendPort(EventSendPort):

    def __init__(self, name):
        self._name = name
        self.receivers = []

    def update(self, events):
        for event in events:
            self.send(event)

    @property
    def name(self):
        return self._name


class NetworkGraph(object):

    def __init__(self, nodes=None):
        if nodes is None:
            self._nodes = {}
        else:
            self._nodes = {n.key: n for n in nodes}

    def __contains__(self, node):
        return node.key in self._nodes

    def __len__(self):
        return len(self._nodes)

    def __repr__(self):
        return pformat(list(self.nodes))

    @property
    def nodes(self):
        if PY2:
            return self._nodes.itervalues()
        else:
            return self._nodes.values()

    def node(self, comp_array, index):
        return self._nodes[(comp_array, index)]

    @property
    def edges(self):
        return chain(*(n.out_edges for n in self.nodes))

    def add_node(self, comp_array, index, sample_index, props, rank):
        node = Node(comp_array, index, sample_index, props, rank)
        self._nodes[node.key] = node
        return node

    def add_sink(self, comp_array, index, src_port, communicates, delay):
        sink = SinkNode()
        self.add_edge(self.node(comp_array, index), sink, src_port, None,
                      communicates, delay)
        self._nodes[sink.key] = sink
        return sink

    def add_source(self, comp_array, index, signal, dest_port, communicates,
                   delay):
        source = SourceNode(signal)
        self.add_edge(self.node(comp_array, index), source, None,
                      dest_port, communicates, delay)
        self._nodes[source.key] = source
        return source

    def add_edge(self, src_node, dest_node, src_port, dest_port, communicates,
                 delay):
        # Use weak refs to reference nodes and edges to allow them to be
        # garbage collected
        edge = Edge(src_node, dest_node, src_port, dest_port, communicates,
                    delay)
        src_node.out_edges.append(edge)
        dest_node.in_edges.append(edge)
        return edge

    def remove_nodes(self, nodes):
        # Delete nodes from dictionary
        for node in nodes:
            # Remove node from graph dict
            del self._nodes[node.key]
            # Delete edges connected to/from this node
            for edge in node.in_edges:
                edge.src_node.out_edges.remove(edge)
            for edge in node.out_edges:
                edge.dest_node.in_edges.remove(edge)


Edge = namedtuple('Edge',
                  'src_node dest_node src_port dest_port communicates delay')


class Node(object):

    type = 'node'

    def __init__(self, comp_array, index, sample_index, props, rank):
        self.comp_array = comp_array
        self.index = index
        self.sample_index = sample_index
        self.props = props
        self.rank = rank
        self.out_edges = []
        self.in_edges = []
        self.remote_send_ports = []
        self.remote_receive_ports = {}

    def __repr__(self):
        return 'Node({}, {}, {}, {})'.format(self.comp_array, self.index,
                                             self.sample_index, self.props)

    @property
    def edges(self):
        return chain(self.out_edges, self.in_edges)

    def unconnected(self):
        return Node(
            self.comp_array,
            self.index,
            self.sample_index,
            self.props,
            self.rank)

    @property
    def key(self):
        return (self.comp_array, self.index)

    @property
    def degree(self):
        return len(self.in_edges) + len(self.out_edges)


class SourceNode(object):

    type = 'source'

    def __init__(self, signal):
        self.signal = signal
        self.out_edges = []
        self.remote_send_ports = []
        self.remote_receive_ports = {}

    def __repr__(self):
        return "Source('{}')".format(self.key)

    @property
    def key(self):
        edge = self.edge
        return edge.dest_node.key + (edge.dest_port,)

    @property
    def degree(self):
        return len(self.out_edges)

    @property
    def edge(self):
        assert len(self.out_edges) == 1, "Connect Source before accessing key"
        return self.out_edges[0]

    @property
    def connected_to(self):
        return self.edge.dest_node


class SinkNode(object):

    type = 'sink'

    def __init__(self):
        self.in_edges = []
        self.remote_send_ports = []
        self.remote_receive_ports = {}

    def __repr__(self):
        return "Sink('{}')".format(self.key)

    @property
    def key(self):
        edge = self.edge
        return edge.src_node.key + (edge.src_port,)

    @property
    def edge(self):
        assert len(self.in_edges) == 1, "Connect Sink before accessing key"
        return self.in_edges[0]

    @property
    def connected_to(self):
        return self.edge.src_node

    @property
    def degree(self):
        return len(self.in_edges)

    @property
    def out_edges(self):
        return []
