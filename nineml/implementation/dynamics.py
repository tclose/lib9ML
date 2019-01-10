from functools import reduce  # Required for Python 3
from operator import itemgetter
from itertools import chain
from copy import copy
from collections import deque, OrderedDict
import bisect
import math
import numpy as np
import sympy as sp
import nineml.units as un
from nineml.exceptions import NineMLUsageError, NineMLNameError


class Dynamics(object):
    """
    Representation of a Dynamics object
    """

    def __init__(self, component_class, properties, initial_state, start_t,
                 regime_kwargs=None, **kwargs):
        if properties.component_class != component_class:
            raise NineMLUsageError(
                "Provided properties do not match defn ({} and {})"
                .format(properties, component_class))
        if initial_state.component_class != component_class:
            raise NineMLUsageError(
                "Provided state does not match defn ({} and {})"
                .format(initial_state, component_class))
        # Recursively substitute and remove all aliases that are not referenced
        # in analog-send-ports
        self.defn = component_class.substitute_aliases()
        # Convert all quantities to SI units for simplicity
        properties = properties.in_si_units()
        initial_state = initial_state.in_si_units()
        start_t = float(start_t.in_units(un.s))
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
        for port_defn in self.defn.event_send_ports:
            self.event_send_ports[port_defn.name] = AnalogSendPort(
                port_defn, self)
        for port_defn in self.defn.analog_send_ports:
            self.analog_send_ports[port_defn.name] = AnalogSendPort(
                port_defn, self)
        for port_defn in self.defn.event_receive_ports:
            self.event_receive_ports[port_defn.name] = EventReceivePort(
                port_defn, self)
        for port_defn in self.defn.analog_receive_ports:
            self.analog_receive_ports[port_defn.name] = AnalogReceivePort(
                port_defn, self)
        for port_defn in self.defn.analog_reduce_ports:
            self.analog_reduce_ports[port_defn.name] = AnalogReducePort(
                port_defn, self)
        # Initialise regimes
        self.regimes = {}
        for regime_def in self.defn.regimes:
            if regime_kwargs is not None:
                kwgs = regime_kwargs[regime_def.name]
            else:
                kwgs = kwargs
            try:
                regime = LinearRegime(regime_def, self, **kwgs)
            except SolverNotValidForRegimeException:
                regime = NonlinearRegime(regime_def, self, **kwgs)
            self.regimes[regime_def.name] = regime
        self.current_regime = self.regimes[initial_state.regime]

    def simulate(self, stop_t):
        stop_t = float(stop_t.in_units(un.s))
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
        # Add special symbols
        values = {'t': t, 'pi': math.pi}
        values.update(state)
        values.update((n, p.value(t))
                      for n, p in chain(self.analog_receive_ports.items(),
                                        self.analog_reduce_ports.items()))
        values.update(self.parameters.items())
        try:
            value = values[expr]  # For simple exprs, which are just a var name
        except KeyError:
            try:
                value = float(expr.evalf(subs=values))
            except TypeError:
                value = bool(expr.subs(values))
        return value

    def update_state(self, new_state, new_t):
        self._state = new_state
        self._t = new_t
        for port in chain(self.analog_send_ports.values(),
                          self.event_receive_ports.values()):
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

    def __init__(self, defn, parent):
        self.defn = defn
        self.parent = parent
        self.on_conditions = []
        for oc_def in self.defn.on_conditions:
            self.on_conditions.append(OnCondition(oc_def, self))
        self.on_events = []
        for oe_def in self.defn.on_events:
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
        while self.parent.t < stop_t:
            # If new state has been assigned by a transition in a previous
            # iteration then we don't do an ODE step to allow for multiple
            # transitions at the same time-point
            if transition is None:
                # Attempt to step to the end of the update window or until the
                # next incoming event. Although in practice the step-size of
                # the solver should be small enough for acceptable levels of
                # accuracy and so that the time of trigger-activations can be
                # approximated by linear intersections.
                max_dt = (min(self.time_of_next_handled_event, stop_t) -
                          self.parent.t)
                proposed_state, proposed_t = self.step_odes(max_dt=max_dt)
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

    def step_odes(self, max_dt=None):
        # Implemented in sub-classes
        raise NotImplementedError

    @property
    def time_of_next_handled_event(self):
        try:
            return min(oe.port.next_event for oe in self.on_events)
        except ValueError:
            return float('inf')

    @property
    def name(self):
        return self.defn.name


class Transition(object):

    def __init__(self, defn, parent):
        self.defn = defn
        self.parent = parent

    @property
    def output_event_names(self):
        return self.defn.output_event_names


class OnCondition(Transition):

    def __init__(self, defn, parent):
        Transition.__init__(self, defn, parent)
        self.trigger_expr = self.defn.trigger.rhs
        if self.defn.trigger.crossing_time_expr is not None:
            self.trigger_time_expr = (
                self.defn.trigger.crossing_time_expr.rhs)
        else:
            self.trigger_time_expr = None

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
        if not dynamics.eval(self.trigger_expr) and dynamics.eval(
                self.trigger_expr, state=proposed_state, t=proposed_t):
            if self.trigger_time_expr is not None:
                trigger_time = dynamics.eval(self.trigger_time_expr)
            else:
                # Default to proposed time if we can't solve trigger expression
                trigger_time = proposed_t
        else:
            trigger_time = float('inf')
        return trigger_time


class OnEvent(Transition):

    def __init__(self, defn, parent):
        Transition.__init__(self, defn, parent)
        self.port = self.parent.parent.event_receive_ports[
            self.defn.name]


class Port(object):

    def __init__(self, defn, parent):
        self.defn = defn
        self.parent = parent

    @property
    def name(self):
        return self.defn.name


class EventSendPort(Port):

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.receivers = []

    def send(self, t):
        for receiver, delay in self.receivers:
            receiver.receive(t + delay)

    def connect_to(self, receive_port, delay):
        self.receivers.append((receive_port, delay))


class EventReceivePort(Port):

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
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

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.receivers = []
        # A list that saves the value of the send port in a buffer at
        self.buffer = deque()
        # Get the expression that defines the value of the port
        try:
            self.expr = self.parent.defn.alias(self.name).rhs.subs(
                self.parent.parent.parameters)
        except NineMLNameError:
            self.expr = self.name

    def values(self, times):
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
        rec_times, values = np.array(self.buffer).T
        return np.interp(times, rec_times, values)

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

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.sender = None
        self.delay = None

    def connect_from(self, send_port, delay):
        if self.sender is not None:
            raise NineMLUsageError(
                "Cannot connect analog receive port '{}' in {} to multiple "
                "receive ports")
        self.sender = send_port
        self.delay = float(delay.in_units(un.s))

    def value(self, t):
        return self.values([t])[0]

    def values(self, times):
        if self.sender is None:
            raise NineMLUsageError(
                "Analog receive port '{}' in {} has not been connected"
                .format(self.name, self.parent))
        return self.sender.values(t - self.delay for t in times)


class AnalogReducePort(Port):

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.senders = []

    @property
    def operator(self):
        return self.defn.python_op

    def connect_from(self, send_port, delay):
        self.senders.append((send_port, float(delay.in_units(un.s))))

    def value(self, t):
        return self.values([t])[0]

    def values(self, times):
        return reduce(self.operator, (sender.values([t - delay for t in times])
                                      for sender, delay in self.senders))


class AnalogSource(object):
    """
    An input source that can be connected to an AnalogReceivePort or
    AnalogSendPort
    """

    def __init__(self, signal):
        self.buffer = list(signal)

    def connect_to(self, receive_port, delay):
        receive_port.connect_from(self, delay)

    def value(self, t):
        return self.values([t])[0]

    def values(self, times):
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
        rec_times, values = np.array(self.buffer).T
        return np.interp(times, rec_times, values)

    @classmethod
    def step(cls, amplitude, start_time, duration, dt, delay):
        start_time = float(start_time.in_units(un.s))
        duration = float(duration.in_units(un.s))
        dt = float(dt.in_units(un.s))
        delay = float(delay.in_units(un.s))
        start_minus_delay = start_time - delay
        num_preceding = int(np.floor(start_minus_delay / dt))
        num_remaining = int(np.ceil((duration - start_minus_delay) / dt))
        amplitude = amplitude.in_si_units()
        signal = np.concatenate((np.zeros(num_preceding),
                                 (np.ones(num_remaining) * float(amplitude))))
        times = np.arange(start_time, start_time + duration, dt)
        return AnalogSource(zip(times, signal))


class EventSource(object):

    def __init__(self, events):
        self.events = list(events)

    def connect_to(self, receive_port, delay):
        for event_t in self.events:
            receive_port.receive(event_t + delay)


class LinearRegime(Regime):
    """
    Extends the base Regime class to implements a solver for a set of linear
    ODES. The solver is adapted from Brian2's "exact" state updater module.

    <brian2-cite-to-go-here>

    Parameters
    ----------
    defn : nineml.Regime
        The 9ML defn of the regime
    parent : Dynamics
        The dynamics object containing the current regime
    dt : nineml.Quantity (time)
        The default step size for the ODE updates
    """

    def __init__(self, defn, parent, dt, **kwargs):  # @UnusedVariable
        Regime.__init__(self, defn, parent)
        self.dt = float(dt.in_units(un.s))

        # Convert state variable names into Sympy symbols
        active_vars = [
            sp.Symbol(n) for n in self.defn.time_derivative_variables]

        if not active_vars:
            self.default_update_exprs = None
        else:

            # Get coefficient matrix
            self.coeffs = sp.zeros(len(active_vars))
            constants = sp.zeros(len(active_vars), 1)

            # Populate matrix of coefficients
            for i, sv_row in enumerate(active_vars):
                try:
                    time_deriv = self.defn.time_derivative(str(sv_row))
                except NineMLNameError:
                    # No time derivative for the state variable in this regime
                    # so the matrix/vector rows should stay zero
                    continue
                # Iterate over all state variables one at a time and detect
                # their coefficients. Each iteration removes a state variable
                # from the expression until all that is left is the constant
                expr = time_deriv.rhs.expand()

                for j, sv_col in enumerate(active_vars):
                    expr = expr.collect(sv_col)
                    coeff_wildcard = sp.Wild('_k', exclude=active_vars)
                    constant_wildcard = sp.Wild('_c', exclude=[sv_col])
                    pattern = coeff_wildcard * sv_col + constant_wildcard
                    match = expr.match(pattern)
                    if match is None:
                        raise SolverNotValidForRegimeException
                    self.coeffs[i, j] = match[coeff_wildcard]
                    expr = match[constant_wildcard]

                constants[i] = expr

            solution = sp.solve_linear_system(self.coeffs.row_join(constants),
                                              *active_vars)
            if solution is None or sorted(solution.keys()) != active_vars:
                raise SolverNotValidForRegimeException

            self.b = sp.ImmutableMatrix(
                [solution[s] for s in active_vars]).transpose()

            self.default_update_exprs = self._update_exprs(self.dt)

    def _update_exprs(self, dt):

        # Solve the system
        try:
            A = (self.coeffs * dt).exp()
        except NotImplementedError:
            raise SolverNotValidForRegimeException
        C = sp.ImmutableMatrix([A.dot(self.b)]) - self.b
        _S = sp.MatrixSymbol('_S', self.defn.num_time_derivatives, 1)
        updates = A * _S + C.transpose()
        updates = updates.as_explicit()

        update_exprs = {}

        for td_var, update in zip(self.defn.time_derivative_variables,
                                  updates):
            expr = update
            if len(expr.atoms(sp.I)) > 0:
                raise SolverNotValidForRegimeException
            for i, state_var in enumerate(self.defn.time_derivative_variables):
                expr = expr.subs(_S[i, 0], state_var)
            update_exprs[td_var] = expr
        return update_exprs

    def step_odes(self, max_dt=None):  # @UnusedVariable
        if max_dt < self.dt:
            update_exprs = (self.update_exprs(max_dt)
                            if self.default_update_exprs is not None else None)
            proposed_t = self.parent.t + max_dt
        else:
            update_exprs = self.default_update_exprs
            proposed_t = self.parent.t + self.dt
        if update_exprs is not None:
            proposed_state = copy(self.parent.state)
            proposed_state.update(
                (n, self.parent.eval(e)) for n, e in update_exprs.items())
        else:
            proposed_state = self.parent.state
        return proposed_state, proposed_t


class NonlinearRegime(Regime):

    def __init__(self, defn, parent, dt, **kwargs):  # @UnusedVariable
        Regime.__init__(self, defn, parent)


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
