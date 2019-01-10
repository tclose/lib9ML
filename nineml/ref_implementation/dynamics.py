from functools import reduce  # Required for Python 3
from operator import itemgetter
from itertools import chain
from collections import deque, OrderedDict
import bisect
import math.pi
import numpy as np
import sympy as sp
import nineml.units as un
from nineml.exceptions import NineMLUsageError, NineMLNameError


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
        # Convert all quantities to SI units for simplicity
        properties = properties.in_si_units()
        initial_state = initial_state.in_si_units()
        start_t = start_t.in_units(un.s)
        # Initialise state information
        self._parameters = {p.name: float(p.value)
                            for p in properties.properties}
        self._state = OrderedDict((sv.name, float(sv.value))
                                  for sv in initial_state.variables)
        self._t = start_t
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
            try:
                regime = LinearRegime(regime_def, self)
            except SolverNotValidForRegimeException:
                regime = NonlinearRegime(regime_def, self)
            self.regimes[regime_def.name] = regime
        self._current_regime = self.regimes[initial_state.regime_name]

    def simulate(self, stop_t, incoming_events=None):
        # Register incoming events, with the closest on top, only used
        # when simulating the dynamics object independently, i.e. not in
        # a network where the events are passed directly between ports
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

    def eval(self, expr, state=None, t=None):
        """
        Evaluates the given expression given the values of the state
        and ports of the Dynamics object. Returns a dictionary mapping variable
        (parameter| port) names to their current values.

        Parameters
        ----------
        expr : sympy.Expression
            The expression to evaluate
        state : numpy.array
            The value of the state. If None the current state is used
        t : float
            The value of the time. If None the current time is used

        Returns
        -------
        value : float
            The value of the expression
        """
        if state is None:
            state = self.state
        if t is None:
            t = self.t
        values = {'t': t}
        values.update(state)
        values.update((p.name, p.value(t))
                      for p in chain(self.analog_receive_ports,
                                     self.analog_reduce_ports))
        try:
            value = values[expr]  # For simple exprs, which are just a var name
        except KeyError:
            value = float(expr.evalf(values))
        return value

    def sub_parameters(self, expr):
        return expr.subs(chain(self.parameters.items(), [('pi', math.pi)]))

    def update_state(self, new_state, new_t):
        self.state_array = new_state
        self.t = new_t
        for port in chain(self.analog_send_ports, self.event_receive_ports):
            port.update_buffer()

    @property
    def parameters(self):
        return self._parameters

    @property
    def t(self):
        return self._t

    @property
    def state(self):
        return self._state

    def clear_buffers(self, min_t):
        for port in self.analog_send_ports:
            port.clear_buffer(min_t)


class Regime(object):
    """

    Simulate the regime for the given duration unless an event is
    raised.

    Parameters
    ----------
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
        stop_t : Quantity(time)
            The time to run the simulation until
        """
        transition = None
        while self.t < stop_t:
            # If new state has been assigned by a transition in a previous
            # iteration then we don't do an ODE step to allow for multiple
            # transitions at the same time-point
            if transition is None:
                # Attempt to step to the end of the update window or until the
                # next incoming event. Although in practice the step-size of
                # the solver should be small enough for acceptable levels of
                # accuracy and so that the time of trigger-activations can be
                # approximated by linear intersections.
                max_step = (min(self.time_of_next_handled_event, stop_t) -
                            self.t)
                proposed_state, proposed_t = self.step_odes(max_step=max_step)
            # Find next transition that will occur (on-conditions that aren't
            # triggered or ports with no upcoming events are set to a time of
            # 'inf').
            transitions = [(oc, oc.time_of_trigger(proposed_state, proposed_t))
                           for oc in self.on_conditions]
            transitions.extend((oe, oe.port.time_of_next_event)
                               for oe in self.on_events)
            try:
                transition, transition_t = min(transitions, key=itemgetter(1))
            except ValueError:
                transition = None  # If there are no transitions in the regime
            else:
                if transition_t > proposed_t:
                    # The next transition doesn't occur before the end of the
                    # current step so can be ignored
                    transition = None
            new_regime = None
            if transition is not None:
                # Action the transition assignments and output events
                if transition.has_state_assignments():
                    proposed_state = transition.assign_states(self.state_array)
                    # If the state has been assigned mid-step, we rewind the
                    # simulation to the time of the assignment
                    proposed_t = transition_t
                for port_name in transition.output_event_names:
                    self.parent.event_send_port[port_name].send(transition_t)
                if transition.target_regime_name != self.name:
                    new_regime = transition.target_regime_name
            # Update the state and buffers
            self.parent.update_state(proposed_state, proposed_t)
            # Transition to new regime if specified in the active transition.
            # NB: that this transition occurs after all other elements of the
            # transition and the update of the state/time
            if new_regime is not None:
                raise RegimeTransition(new_regime)

    def step_odes(self, max_step=None):
        # Implemented in sub-classes
        raise NotImplementedError

    @property
    def time_of_next_handled_event(self):
        return min(oe.port.next_event for oe in self.on_events.values())


class Transition(object):

    def __init__(self, definition, parent):
        self.definition = definition
        self.parent = parent

    @property
    def output_event_names(self):
        return self.definition.output_event_names


class OnCondition(Transition):

    def __init__(self, definition, parent):
        Transition.__init__(self, definition, parent)
        dynamics = parent.parent
        self.trigger = dynamics.sub_parameters(self.definition.trigger.rhs)
        self.trigger_time = dynamics.sub_parameters(
            self.definition.trigger.crossing_time_expr.rhs)

    def time_of_trigger(self, proposed_state, proposed_t):
        """
        Checks to see if the condition will be triggered by the proposed
        state and time

        Parameters
        ----------
        proposed_state : numpy.array(float)
            The proposed values for the state variables
        proposed_t : float
            The time at which the values are proposed to be updated

        Returns
        -------
        trigger_time : float
            The exact time point between the current time and the proposed new
            time that the condition was triggered. If the condition was not
            triggered then it returns 'inf'
        """
        dynamics = self.parent.parent
        if (dynamics.eval(self.trigger, state=proposed_state, t=proposed_t) and
                not dynamics.eval(self.trigger_time)):
            if self.trigger_time is not None:
                trigger_time = dynamics.eval(self.trigger_time)
            else:
                # Default to proposed time if we can't solve trigger expression
                trigger_time = proposed_t
        else:
            trigger_time = float('inf')
        return trigger_time


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

    def update_buffer(self):
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
        self.buffer = deque()
        # Get the expression that defines the value of the port
        try:
            self.expr = self.parent.definition.alias(self.name).rhs.subs(
                self.parent.parent.parameters)
        except NineMLNameError:
            self.expr = self.name

    def value(self, times):
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
        return np.interp(times, *np.array(self.buffer).T)

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
            self.buffer.append((self.parent.t, self.parent.eval(self.expr)))

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

    def value(self, times):
        if self.sender is None:
            raise NineMLUsageError(
                "Analog receive port '{}' in {} has not been connected"
                .format(self.name, self.parent))
        return self.sender.value(t - self.delay for t in times)


class AnalogReducePort(Port):

    def __init__(self, definition, parent):
        Port.__init__(self, definition, parent)
        self.senders = []

    @property
    def operator(self):
        return self.definition.python_op

    def connect_from(self, send_port, delay):
        self.senders.append((send_port, delay))

    def value(self, times):
        return reduce(self.operator, (sender.value(t - delay for t in times)
                                      for sender, delay in self.senders))


class LinearRegime(Regime):
    """
    Extends the base Regime class to implements a solver for a set of linear
    ODES. The solver is adapted from Brian2's "exact" state updater module.

    <brian2-cite-to-go-here>
    """

    def __init__(self, definition, parent, dt):
        Regime.__init__(definition, parent)

        # Convert state variable names into Sympy symbols
        state_vars = [
            sp.Symbol(n) for n in self.parent.definition.state_variable_names]

        # Get coefficient matrix
        coeffs = sp.zeros(len(state_vars))
        constants = sp.zeros(len(state_vars), 1)

        # Populate matrix of coefficients
        for i, sv_row in enumerate(state_vars):
            try:
                time_deriv = self.definition.time_derivative(str(sv_row))
            except NineMLNameError:
                # No time derivative for the state variable in this regime
                # so the matrix/vector rows should stay zero
                continue
            # Iterate over all state variables one at a time and detect their
            # coefficients. Each iteration removes a state variable from the
            # expression until all that is left is the constant
            expr = self.parent.sub_parameters(time_deriv.rhs)

            for j, sv_col in enumerate(state_vars):
                expr = expr.collect(sv_row)
                coeff_wildcard = sp.Wild('__k__', exclude=state_vars)
                constant_wildcard = sp.Wild('__c__', exclude=[sv_col])
                pattern = coeff_wildcard * sv_col + constant_wildcard
                match = expr.match(pattern)
                if match is None:
                    raise SolverNotValidForRegimeException
                coeffs[i, j] = match[coeff_wildcard]
                expr = match[constant_wildcard]

        # The remaining expression should be numeric
        try:
            float(expr)
        except ValueError:
            raise SolverNotValidForRegimeException
        constants[i] = expr

        solution = sp.solve_linear_system(coeffs.row_join(constants),
                                          *state_vars)
        if solution is None or sorted(solution.keys()) != state_vars:
            raise SolverNotValidForRegimeException

        b = sp.ImmutableMatrix(solution[s] for s in state_vars).transpose()

        # Solve the system
        try:
            self.A = (coeffs * dt).exp()
        except NotImplementedError:
            raise SolverNotValidForRegimeException
        self.C = sp.ImmutableMatrix([self.A.dot(b)]) - b

    def step_odes(self, max_step=None):
        
        


class NonlinearRegime(Regime):

    def step_odes(self, max_step=None):
        pass


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


class SolverNotValidForRegimeException(Exception):
    """
    Raised when the selected solver isn't appropriate for the given ODE
    system
    """
    pass
