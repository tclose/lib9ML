from functools import reduce  # Required for Python 3
from operator import itemgetter
import bisect
import numpy as np
from nineml.exceptions import NineMLUsageError


class Dynamics(object):
    """
    Representation of a Dynamics object
    """

    def __init__(self, definition, properties, initial_state, start_t):
        if properties.definition != definition:
            raise NineMLUsageError(
                "Provided properties do not match definition ({} and {})"
                .format(properties, definition))
        if initial_state.definition != definition:
            raise NineMLUsageError(
                "Provided state does not match definition ({} and {})"
                .format(initial_state, definition))
        # Recursively substitute and remove all aliases that are not referenced
        # in analog-send-ports
        self.definition = definition.subsitute_aliases()
        # Initialise ports
        self.event_send_ports = {}
        self.analog_send_ports = {}
        self.event_receive_ports = {}
        self.analog_receive_ports = {}
        self.analog_reduce_ports = {}
        for port_def in self.definition.event_send_ports:
            self.event_send_ports[port_def.name] = AnalogSendPort(
                port_def, self)
        for port_def in self.definition.analog_send_ports:
            self.analog_send_ports[port_def.name] = AnalogSendPort(
                port_def, self)
        for port_def in self.definition.event_receive_ports:
            self.event_receive_ports[port_def.name] = EventReceivePort(
                port_def, self)
        for port_def in self.definition.analog_receive_ports:
            self.analog_receive_ports[port_def.name] = AnalogReceivePort(
                port_def, self)
        for port_def in self.definition.analog_reduce_ports:
            self.analog_reduce_ports[port_def.name] = AnalogReducePort(
                port_def, self)
        # Initialise regimes
        self.regimes = {}
        for regime_def in self.definition.regimes:
            if regime_def.is_linear:
                regime = LinearRegime(regime_def, self)
            else:
                regime = NonlinearRegime(regime_def, self)
            self.regimes[regime_def.name] = regime
        # Initialise state information
        self.current_regime = self.regimes[initial_state.regime_name]
        self.state_array = np.copy(initial_state.variables)
        self.t = start_t

    def simulate(self, stop_t, incoming_events=None):
        """
        """
        # Create stack of upcoming events, with the closest on top
        for port_name, t in incoming_events:
            self.event_receive_ports[port_name].receive(t)
        # Update the simulation until stop_t
        while self.t < stop_t:
            # Simulte the current regime until t > stop_t or there is a
            # transition to a new regime
            try:
                self.current_regime.simulate(stop_t)
            except RegimeTransition as transition:
                self.current_regime = self.regimes[transition.target]

    def value(self, name):
        


class Regime(object):
    """

    Simulate the regime for the given duration unless an event is
    raised.

    Parameters
    ----------
    solver : Solver
        A solver object, which runs the state updates
    """

    def __init__(self, definition, parent):
        self.definition = definition
        self.parent = parent
        self.on_conditions = []
        for oc_def in self.definition.on_conditions:
            self.on_conditions.append(OnCondition(oc_def, self))
        self.on_events = []
        for oe_def in self.definition.on_events:
            self.on_events.append(OnEvent(oe_def, self))

    def simulate(self, stop_t):
        """
        Simulate the regime for the given duration unless an event is
        raised.

        Parameters
        ----------
        start_t : Quantity(time)
            The time to run the simulation from
        stop_t : Quantity(time)
            The time to run the simulation until
        state : dict[str, Quantity]
            The initial state of the simulation
        solver : Solver
            A solver object, which runs the state updates
        """
        transition = None
        while self.t < stop_t:
            # If new state has been assigned by a transition in a previous
            # iteration then we don't do an ODE step to allow for multiple
            # transitions at the same time-point
            if not transition:
                # Attempt to step to the end of the update window or until the
                # next incoming event. Although in practice the step-size of
                # the solver should be small enough for acceptable levels of
                # accuracy and so that the time of trigger-activations can be
                # approximated by linear intersections.
                max_step = (min(self.time_of_next_handled_event, stop_t) -
                            self.t)
                proposed_state, proposed_t = self.step_odes(max_step=max_step)
            # Find next occuring transition (on-conditions that aren't
            # triggered or ports with no upcoming events are set to a time of
            # 'inf')
            transitions = [(oc.time_of_trigger(proposed_state, proposed_t), oc)
                           for oc in self.on_conditions]
            transitions.extend((oe.port.time_of_next_event, oe)
                               for oe in self.on_events)
            transition_t, transition = min(transitions, key=itemgetter(0))
            if transition_t > proposed_t:
                # The next transition doesn't occur before the end of the
                # current step so can be ignored
                transition = None
            new_regime = None
            if transition:
                # Action the transition assignments and output events
                if transition.has_state_assignments():
                    proposed_state = transition.assign_states(self.state_array)
                    proposed_t = transition_t
                for port_name in transition.output_event_names:
                    self.parent.event_send_port[port_name].send(transition_t)
                if transition.target_regime_name != self.name:
                    new_regime = transition.target_regime_name
            # Update the state and buffers
            self.state_array = proposed_state
            self.t = proposed_t
            for port in self.parent.analog_send_ports:
                port.update_buffer()
            # Transition to new regime if specified in the active transition.
            # Note that this occurs after all other elements of the transition
            # and the update of the state/time
            if new_regime is not None:
                raise RegimeTransition(new_regime)

    def step_odes(self, max_step=None):
        # Implemented in sub-classes
        raise NotImplementedError

    @property
    def time_of_next_handled_event(self):
        return min(oe.port.next_event for oe in self.on_events.values())

    @property
    def state_array(self):
        return self.parent.state_array

    @state_array.setter
    def state_array(self, array):
        self.parent.state_array = array

    @property
    def t(self):
        return self.parent.t

    @t.setter
    def t(self, t):
        self.parent.t = t


class LinearRegime(Regime):

    def step_odes(self, max_step=None):
        pass


class NonlinearRegime(Regime):

    def step_odes(self, max_step=None):
        pass


class Transition(object):

    def __init__(self, definition, parent):
        self.definition = definition
        self.parent = parent

    @property
    def output_event_names(self):
        return self.definition.output_event_names


class OnCondition(Transition):

    def time_of_trigger(self, new_state, new_t):
        pass


class OnEvent(Transition):

    def __init__(self, definition, parent):
        Transition.__init__(self, definition, parent)
        self.port = self.parent.parent.event_receive_ports[
            self.definition.name]


class Port(object):

    def __init__(self, definition, parent):
        self.definition = definition
        self.parent = parent

    @property
    def name(self):
        return self.definition.name


class EventSendPort(Port):

    def __init__(self, definition, parent):
        Port.__init__(self, definition, parent)
        self.receivers = []

    def send(self, t):
        for receiver, delay in self.receivers:
            receiver.receive(t + delay)

    def connect_to(self, receive_port, delay):
        self.receivers.append((receive_port, delay))


class EventReceivePort(Port):

    def __init__(self, definition, parent):
        Port.__init__(self, definition, parent)
        self.events = []

    def receive(self, t):
        bisect.insort(self.events, t)

    def clear_past(self):
        self.events = bisect.bisect(self.events, self.parent.t)

    @property
    def time_of_next_event(self):
        try:
            return self.events[0]
        except IndexError:
            return float('inf')


class AnalogSendPort(Port):

    def __init__(self, definition, parent):
        Port.__init__(self, definition, parent)
        self.receivers = []
        # A list that saves the value of the send port in a buffer at
        self.buffer = []

    def value(self, t):
        """
        Returns value of the buffer linearly interpolated to the requested
        time

        Parameters
        ----------
        times : np.array(float)
            The times (in seconds) to return the value of the send port at

        Returns
        -------
        values : np.array(float)
            The values of the state-variable/alias that the send port
            references interpolated to the given time points
        """
        return np.interp([t], *np.array(self.buffer).T)

    def connect_to(self, receive_port, delay):
        # Register the sending port with the receiving port so it can retrieve
        # the values of the sending port
        receive_port.connect_from(self, delay)
        # Keep track of the receivers connected to this send port
        self.receivers.append(receive_port)

    def update_buffer(self):
        """
        Buffers the value of the port for reference by receivers
        """
        if self.receivers:
            self.buffer.append((self.parent.t.in_SI(),
                                self.parent.value(self.name)))

    def clear_buffer(self, min_t):
        while self.buffer[0][0] < min_t:
            self.buffer.popleft()


class AnalogReceivePort(Port):

    def __init__(self, definition, parent):
        Port.__init__(self, definition, parent)
        self.sender = None
        self.delay = None

    def connect_from(self, send_port, delay):
        if self.sender is not None:
            raise NineMLUsageError(
                "Cannot connect analog receive port '{}' in {} to multiple "
                "receive ports")
        self.sender = send_port
        self.delay = delay

    @property
    def value(self):
        if self.sender is None:
            raise NineMLUsageError(
                "Analog receive port '{}' in {} has not been connected"
                .format(self.name, self.parent))
        return self.sender.value(self.parent.t - self.delay)


class AnalogReducePort(Port):

    def __init__(self, definition, parent):
        Port.__init__(self, definition, parent)
        self.senders = []

    @property
    def operator(self):
        return self.definition.python_op

    def connect_from(self, send_port, delay):
        self.senders.append((send_port, delay))

    @property
    def value(self):
        return reduce(self.operator,
                      (s.value(self.parent.t - d) for s, d in self.senders))


class RegimeTransition(Exception):
    """
    Raised when a transition to a new regime is triggered

    Parameters
    ----------
    target : str
        Name of the regime to transition to
    """
    def __init__(self, target):
        self.target = target
