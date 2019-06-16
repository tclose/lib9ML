from functools import reduce  # Required for Python 3
from operator import itemgetter
from itertools import chain
from copy import copy
from collections import deque, OrderedDict
import numpy.random
import bisect
from logging import getLogger
import sympy as sp
from tqdm import tqdm
from nineml.exceptions import (
    NineMLUsageError, NineMLNameError)
from nineml.units import Quantity
from nineml.user.multi import split_namespace


logger = getLogger('nineml')


# Set common symbol used to denote step size
dt_symbol = 'dt_'


class Dynamics(object):
    """
    An implementation of a Dynamics model in pure Python. The dynamics model
    is flattened into a single-component dynamics class.

    Parameters
    ----------
    model : nineml.user.DynamicsProperties
        The DynamicsProperties to initialise the component with.
    start_t : Quantity (time) | float (s)
        The initial time that dynamics is initialised with
    initial_state : dict(str, float)
        The state that dynamics is initialised with
    """

    def __init__(self, model, start_t, initial_state=None, sample_index=None,
                 initial_regime=None, dynamics_class=None, name=None,
                 rank=None):

        def sample_quantity(elem):
            # If sample index is a dictionary containing different indices for
            # different sub-components
            if isinstance(sample_index, dict):
                index = sample_index[split_namespace(elem.name)[1]]
            else:
                index = sample_index
            if isinstance(elem, Quantity):
                qty = elem
            else:
                qty = elem.quantity
            return float(qty.sample(index).in_si_units())

        if dynamics_class is None:
            dynamics_class = DynamicsClass(model.component_class)
        self.dynamics_class = dynamics_class
        self.model = model
        self.name = name
        self.defn = dynamics_class.defn
        # Initialise state information converting all quantities to SI units
        # for simplicity
        self.properties = OrderedDict(
            (p, sample_quantity(model.property(p)))
            for p in self.defn.parameter_names)
        # Ensure initial state is ordered by var names so it matches up with
        # order in 'dynamics_class.all_symbols'
        self._state = OrderedDict()
        for sv_name in self.defn.state_variable_names:
            try:
                initial = initial_state[sv_name]
            except (TypeError, KeyError):
                try:
                    initial = model.initial_value(sv_name)
                except NineMLNameError:
                    raise NineMLUsageError(
                        "No initial value provided for '{}'"
                        .format(sv_name))
            self._state[sv_name] = sample_quantity(initial)
        if initial_regime is None:
            initial_regime = model.initial_regime
        self.regime = dynamics_class.regimes[initial_regime]
        # Time
        if isinstance(start_t, Quantity):
            start_t = float(start_t.in_si_units())
        self._t = start_t
        # Initialise ports
        self.event_send_ports = OrderedDict()
        self.analog_send_ports = OrderedDict()
        self.event_receive_ports = OrderedDict()
        self.analog_receive_ports = OrderedDict()
        self.analog_reduce_ports = OrderedDict()
        self.ports = {}
        for pdef in self.defn.event_send_ports:
            self.ports[pdef.name] = self.event_send_ports[pdef.name] = (
                EventSendPort(pdef, self))
        for pdef in self.defn.analog_send_ports:
            self.ports[pdef.name] = self.analog_send_ports[pdef.name] = (
                AnalogSendPort(pdef, self))
        for pdef in self.defn.event_receive_ports:
            self.ports[pdef.name] = self.event_receive_ports[pdef.name] = (
                EventReceivePort(pdef, self))
        for pdef in self.defn.analog_receive_ports:
            self.ports[pdef.name] = self.analog_receive_ports[pdef.name] = (
                AnalogReceivePort(pdef, self))
        for pdef in self.defn.analog_reduce_ports:
            self.ports[pdef.name] = self.analog_reduce_ports[pdef.name] = (
                AnalogReducePort(pdef, self))
        self.progress_bar = None
        self.rank = rank

    @property
    def dt(self):
        try:
            dt = self._dt[self.regime.name]
        except TypeError:
            dt = self._dt
        return dt

    def __repr__(self):
        return "{}(model={})".format(type(self).__name__, self.model)

    def update_state(self, new_state, new_t):
        self._state = new_state
        dt = new_t - self._t
        self._t = new_t
        self._update_buffers()
        self.progress_bar.update(dt)

    def _update_buffers(self):
        for port in self.ports.values():
            port.update_buffer()

    def simulate(self, stop_t, dt, show_progress=True):
        # Convert time values to SI units
        if isinstance(stop_t, Quantity):
            stop_t = float(stop_t.in_si_units())
        if isinstance(dt, Quantity):
            dt = float(dt.in_si_units())
        # Set progress bar
        self.progress_bar = tqdm(
            initial=self.t, total=stop_t, desc=(
                "Simulating '{}' (dt={} s)".format(self.model.name, dt)),
            unit='s (sim)', unit_scale=True,
            disable=not show_progress)
        self._update_buffers()
        # Update the simulation until stop_t
        while self.t < stop_t:
            # Simulte the current regime until t > stop_t or there is a
            # transition to a new regime
            try:
                self.regime.update(self, stop_t, dt)
            except RegimeTransition as transition:
                logger.debug("Transitioning from '{}' regime to '{}'"
                             .format(self.regime.name, transition.target))
                self.regime = self.dynamics_class.regimes[transition.target]
        self.progress_bar.close()

    def all_values(self, dt=None, state=None, t=None):
        if state is None:
            state = self.state
        if t is None:
            t = self.t
        all_values = list(chain(
            state.values(),
            (p.value(t) for p in chain(self.analog_receive_ports.values(),
                                       self.analog_reduce_ports.values())),
            self.properties.values(),
            self.dynamics_class.constants.values(),
            [t, dt]))
        return all_values

    @property
    def t(self):
        return self._t

    @property
    def state(self):
        return self._state

    def clear_buffers(self, min_t):
        for port in self.analog_send_ports:
            port.clear_buffer(min_t)

    def port(self, port_name):
        return self.ports[port_name]


class DynamicsClass(object):
    """
    Representation of a Dynamics object
    """

    def __init__(self, model, regime_kwargs=None, **kwargs):
        # Recursively substitute and remove all aliases that are not referenced
        # in analog-send-ports
        logger.info("Initialising '{}' class".format(model.name))
        self.defn = model.substitute_aliases()
        self.constants = OrderedDict(
            (c.name, float(c.quantity.in_si_units()))
            for c in self.defn.constants)
        # Initialise regimes
        self.regimes = {}
        for regime_def in self.defn.regimes:
            if regime_kwargs is not None and regime_def.name in regime_kwargs:
                # Add regime specific kwargs to general kwargs
                kwgs = copy(kwargs)
                kwgs.update(regime_kwargs[regime_def.name])
            else:
                kwgs = kwargs
            try:
                regime = LinearRegime(regime_def, self, **kwgs)
            except SolverNotValidForRegimeException:
                regime = NonlinearRegime(regime_def, self, **kwgs)
            self.regimes[regime_def.name] = regime
        self.port_aliases = {}
        for port in self.defn.analog_send_ports:
            try:
                expr = self.defn.alias(port.name).rhs
            except NineMLNameError:
                expr = sp.Symbol(port.name)
            self.port_aliases[port.name] = self.lambdify(expr)
        logger.info("Finished initialising '{}' class".format(model.name))

    def __repr__(self):
        return "{}(definition={})".format(type(self).__name__,
                                          self.defn)

    @property
    def all_symbols(self):
        "Returns a list of all symbols used in the Dynamics class"
        return list(sp.symbols(chain(
            self.defn.state_variable_names,
            self.defn.analog_receive_port_names,
            self.defn.analog_reduce_port_names,
            self.defn.parameter_names,
            self.constants.keys(),
            ['t', dt_symbol])))

    def lambdify(self, expr, extra_symbols=()):
        "Lambdifies a sympy expression, substituting in all values"
        all_symbols = self.all_symbols
        arg_inds = []
        symbols = []
        # Filter symbols actually required to evaluate the expression
        for i, sym in enumerate(all_symbols):
            if sym in expr.free_symbols:
                symbols.append(sym)
                arg_inds.append(i)
        symbols.extend(extra_symbols)
        func = sp.lambdify(symbols, expr, 'math')
        # Save the indices of the symbols to provide as arguments with the
        # lambda function. In Python > 3.7 we can just provide all symbols as
        # arguments but in < 3.7 we can overflow the maximum number of args
        # (255) for large networks.
        func.arg_inds = arg_inds
        return func


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
        self.updates = {}

    def update(self, dynamics, stop_t, dt):
        """
        Simulate the regime for the given duration unless an event is
        raised.

        Parameters
        ----------
        stop_t : Quantity(time)
            The time to run the simulation until
        """
        if not self.updates:
            # Make the time-step infinite if there ae no time derivative
            # updates as there is no need to step incrementally, just go to the
            # next handled event
            dt = float('inf')
        transitions = []
        while dynamics.t < stop_t:
            # If new state has been assigned by a transition in a previous
            # iteration then we don't do an ODE step to allow for multiple
            # transitions at the same time-point
            if not transitions:
                # Attempt to step to the end of the update window or until the
                # next incoming event. Although in practice the step-size of
                # the solver should be small enough for acceptable levels of
                # accuracy and so that the time of trigger-activations can be
                # approximated by linear intersections.
                step_dt = min(
                    min(self.time_of_next_handled_event(dynamics), stop_t) -
                    dynamics.t, dt)
                logger.debug("Stepping dynamics {} s".format(step_dt))
                proposed_state, proposed_t = self.step(dynamics, step_dt)
            # Detect transitions that have occured during the update step (NB:
            # on- conditions that aren't triggered or ports with no upcoming
            # events are set to a time of 'inf').
            transitions = [(oc, oc.time_of_trigger(dynamics, proposed_state,
                                                   proposed_t))
                           for oc in self.on_conditions]
            transitions.extend((oe, oe.time_of_next_event(dynamics))
                               for oe in self.on_events)
            # Sort transitions in order of the occurence
            transitions = sorted(
                (a for a in transitions if a[1] <= proposed_t),
                key=itemgetter(1))
            new_regime = None
            remaining_transitions_valid = True
            for transition, transition_t in transitions:
                logger.debug("Transitioning {}".format(transition))
                # Send output events
                for port_name in transition.defn.output_event_keys:
                    dynamics.event_send_ports[port_name].send(transition_t)
                # Detect regime change
                if transition.defn.target_regime.name != self.name:
                    new_regime = transition.defn.target_regime.name
                    # Remaining transitions won't neccessarily occur as we will
                    # transition to new regime after updating the state
                    remaining_transitions_valid = False
                # Assign new state values
                if transition.defn.num_state_assignments:
                    # If the state is to been assigned mid-step and not all
                    # states with time-derivatives are assigned, we need to
                    # rewind the simulation to the time of the assignment
                    if transition_t < proposed_t and (
                        set(self.defn.time_derivative_variables) -
                            set(transition.defn.state_assignment_variables)):
                        proposed_state, _ = self.step(
                            dynamics, transition_t - dynamics.t)
                    # Assign states
                    proposed_state = transition.assign_states(dynamics,
                                                              t=transition_t)
                    # Since we have assigned new states, remaining detected
                    # transitions are not necessarily valid and will need to
                    # be checked in the next loop
                    remaining_transitions_valid = False
                proposed_t = transition_t
                if not remaining_transitions_valid:
                    break
            # Update the state and buffers
            dynamics.update_state(proposed_state, proposed_t)
            # Transition to new regime if specified in the active transition.
            # NB: that this transition occurs after all other elements of the
            # transition and the update of the state/time
            if new_regime is not None:
                raise RegimeTransition(new_regime)

    def time_of_next_handled_event(self, dynamics):
        try:
            return min(oe.time_of_next_event(dynamics)
                       for oe in self.on_events)
        except ValueError:
            return float('inf')

    @property
    def name(self):
        return self.defn.name

    def step(self, dynamics, dt):  # @UnusedVariable
        # self.updates should be constructed in the regime __init__ and consist
        # of a dictionaries of functions to update each state, which take the
        # output of self.parent.all_values() as input args
        if self.updates:
            proposed_state = copy(dynamics.state)
            values = dynamics.all_values(dt=dt)
            for var_name, update in self.updates.items():
                # Evaluate ODE update given current state values
                proposed_state[var_name] = update(
                    *[values[i] for i in update.arg_inds])
        else:
            # No time derivatives in regime so state stays the same
            proposed_state = dynamics.state
        proposed_t = dynamics.t + dt
        return proposed_state, proposed_t

    def _lambdify_update_exprs(self, update_exprs, state_vector):
        update_exprs = update_exprs.as_explicit()
        updates = {}
        # Convert Sympy expressions into lambda functions of variables
        for td_var, update_expr in zip(self.defn.time_derivative_variables,
                                       update_exprs):
            update_expr = update_expr.subs({
                state_vector[i, 0]: sv
                for i, sv in enumerate(self.defn.time_derivative_variables)})
            updates[td_var] = self.parent.lambdify(update_expr)
        return updates


class Transition(object):

    random_funcs = {'random_uniform_': numpy.random.uniform,
                    'random_binomial_': numpy.random.binomial,
                    'random_poisson_': numpy.random.poisson,
                    'random_exponential_': lambda x: numpy.random.exponential(
                        1.0 / x),
                    'random_normal_': numpy.random.normal}

    def __init__(self, defn, parent):
        self.defn = defn
        self.parent = parent
        self._assigns = {}
        for assign in self.defn.state_assignments:
            expr = assign.rhs
            # NB: once https://github.com/INCF/nineml-spec/issues/31 is
            # implemented it be necessary to do these substitutions
            rand_vars = {}
            for rand_func_name in self.random_funcs:
                # NB: this falls down if there are two random function calls
                # with the same args in the same expression (but #31 will avoid
                # the need to handle this so I am leaving that for now)
                func_wildcard = sp.Function(rand_func_name)(sp.Wild('x'))
                for i, func_call in enumerate(expr.find(func_wildcard)):
                    value_name = '{}{}'.format(rand_func_name, i)
                    args_lambdas = [self.parent.parent.lambdify(a)
                                    for a in func_call.args]
                    rand_vars[value_name] = (self.random_funcs[rand_func_name],
                                             args_lambdas)
                    expr = expr.subs(func_call, value_name)
            self._assigns[assign.name] = (
                self.parent.parent.lambdify(expr,
                                            extra_symbols=rand_vars.keys()),
                rand_vars)

    def assign_states(self, dynamics, t):
        state = copy(dynamics.state)
        all_values = dynamics.all_values(t=t)
        for var_name, (assign, rand_vars) in self._assigns.items():
            rand_values = {}
            for rv_name, (rand_func, args_lambdas) in rand_vars.items():
                args = [al(*[all_values[i] for i in al.arg_inds])
                        for al in args_lambdas]
                rand_values[rv_name] = rand_func(*args)
            state[var_name] = assign(*[all_values[i] for i in assign.arg_inds],
                                     **rand_values)
        return state


class OnCondition(Transition):

    def __init__(self, defn, parent):
        Transition.__init__(self, defn, parent)
        dynamics = self.parent.parent
        self.trigger_expr = dynamics.lambdify(self.defn.trigger.rhs)
        if self.defn.trigger.crossing_time_expr is not None:
            self.trigger_time_expr = dynamics.lambdify(
                self.defn.trigger.crossing_time_expr.rhs)
        else:
            self.trigger_time_expr = None

    def time_of_trigger(self, dynamics, proposed_state, proposed_t):
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
        values = dynamics.all_values()
        trig = self.trigger_expr
        proposed_values = dynamics.all_values(state=proposed_state,
                                              t=proposed_t)
        if not trig(*[values[i] for i in trig.arg_inds]) and trig(*[
          proposed_values[i] for i in trig.arg_inds]):  # @IgnorePep8
            if self.trigger_time_expr is not None:
                trigger_time = self.trigger_time_expr(*[
                    values[i] for i in self.trigger_time_expr.arg_inds])
            else:
                # Default to proposed time if we can't solve trigger expression
                trigger_time = proposed_t
        else:
            trigger_time = float('inf')
        return trigger_time


class OnEvent(Transition):

    def __init__(self, defn, parent):
        Transition.__init__(self, defn, parent)

    def time_of_next_event(self, dynamics):
        return dynamics.event_receive_ports[
            self.defn.src_port_name].time_of_next_event


class Port(object):

    def __init__(self, defn, parent):
        self.defn = defn
        self.parent = parent

    @property
    def name(self):
        return self.defn.name

    def update_buffer(self):
        pass  # default doesn't require updating

    def __repr__(self):
        return "{}(name='{}')".format(type(self).__name__, self.name)


class EventSendPort(Port):

    communicates = 'event'

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.receivers = []

    def send(self, t):
        logger.debug("Sending events from '{}'".format(self.name))
        for receiver, delay in self.receivers:
            receiver.receive(t + delay)

    def connect_to(self, receive_port, delay):
        if isinstance(delay, Quantity):
            delay = float(delay.in_si_units())
        self.receivers.append((receive_port, delay))


class EventReceivePort(Port):

    communicates = 'event'

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.events = []

    def receive(self, t):
        bisect.insort(self.events, t)

    def update_buffer(self):
        self.events = self.events[bisect.bisect(self.events, self.parent.t):]

    @property
    def time_of_next_event(self):
        try:
            return self.events[0]
        except IndexError:
            return float('inf')


class AnalogSendPort(Port):

    communicates = 'analog'

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.receivers = []
        # To save the values of the send port in a buffer
        self.buffer = deque()
        self.max_delay = 0.0  # To determine the required length of the buffer
        # Get the expression that defines the value of the port
        dynamics = self.parent
        self.expr = dynamics.dynamics_class.port_aliases[defn.name]

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
        value : np.array(float)
            The values of the state-variable/alias that the send port
            references interpolated to the given time points
        """
        # Get buffer value immediately before requested time
        try:
            i, (t1, v1) = next(x for x in enumerate(reversed(self.buffer))
                               if x[1][0] <= t)
        except StopIteration:
            raise NineMLUsageError(
                "Requested time {} s is before beginning of the buffer for "
                "{} ({} s)".format(t, self._location, self.buffer[0][0]))
        # For exact time matches return the value
        if t1 == t:
            return v1
        try:
            t2, v2 = self.buffer[len(self.buffer) - i]
        except IndexError:
            raise NineMLUsageError(
                "Requested time {} s is after end of the buffer for ({} s)"
                .format(t, self._location, self.buffer[-1][0]))
        # Linearly interpolate the point between the buffer
        return v1 + (v2 - v1) * (t - t1) / (t2 - t1)

    @property
    def _location(self):
        return "'{}' port in {}".format(self.name, self.parent)

    def connect_to(self, receive_port, delay):
        if isinstance(delay, Quantity):
            delay = float(delay.in_si_units())
        # Register the sending port with the receiving port so it can retrieve
        # the values of the sending port
        receive_port.connect_from(self, delay)
        # Keep track of the receivers connected to this send port
        self.receivers.append(receive_port)
        if delay > self.max_delay:
            self.max_delay = delay

    def update_buffer(self):
        """
        Buffers the value of the port for reference by receivers
        """
        logger.debug("Updating analog buffer of '{}'".format(self.name))
        if self.receivers:
            values = self.parent.all_values()
            self.buffer.append(
                (self.parent.t,
                 self.expr(*[values[i] for i in self.expr.arg_inds])))
            # Drop buffer values that are no longer required
            if len(self.buffer) > 1:
                min_t = self.parent.t - self.max_delay
                try:
                    while self.buffer[1][0] <= min_t:
                        self.buffer.popleft()
                except IndexError:
                    pass  # Only one value left in buffer
            # If receiver is a sink then we need to update the buffer
            for receiver in self.receivers:
                receiver.update_buffer()

    def clear_buffer(self, min_t):
        while self.buffer[0][0] < min_t:
            self.buffer.popleft()

    @property
    def start_t(self):
        return self.buffer[0][0]

    @property
    def stop_t(self):
        return self.buffer[-1][0]


class AnalogReceivePort(Port):

    communicates = 'analog'

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.sender = None
        self.delay = None

    def connect_from(self, send_port, delay):
        if isinstance(delay, Quantity):
            delay = float(delay.in_si_units())
        if self.sender is not None:
            raise NineMLUsageError(
                "Cannot connect {} to multiple receive ports".format(
                    self._location))
        self.sender = send_port
        self.delay = delay

    def value(self, t):
        logger.debug("Receiving analog value at '{}'".format(self.name))
        try:
            return self.sender.value(t - self.delay)
        except AttributeError:
            raise NineMLUsageError(
                "{} has not been connected".format(self._location))

    @property
    def _location(self):
        return "analog receive port '{}' in {}".format(self.name,
                                                       self.parent)

    @property
    def start_t(self):
        return self.sender.start_t

    @property
    def stop_t(self):
        return self.sender.end_t


class AnalogReducePort(Port):

    def __init__(self, defn, parent):
        Port.__init__(self, defn, parent)
        self.senders = []

    @property
    def operator(self):
        return self.defn.python_op

    def connect_from(self, send_port, delay):
        if isinstance(delay, Quantity):
            delay = float(delay.in_si_units())
        self.senders.append((send_port, delay))

    def value(self, t):
        return reduce(self.operator, (sender.value(t - delay)
                                      for sender, delay in self.senders))

    @property
    def start_t(self):
        return min(s.start_t for s in self.senders)

    @property
    def stop_t(self):
        return max(s.stop_t for s in self.senders)


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

    def __init__(self, defn, parent, **kwargs):  # @UnusedVariable
        Regime.__init__(self, defn, parent)

        # Convert state variable names into Sympy symbols
        active_vars = [
            sp.Symbol(n) for n in self.defn.time_derivative_variables]

        if active_vars:
            logger.info("Solving linear ODEs for '{}' regime of '{}' class"
                        .format(self.defn.name, self.parent.defn.name))
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
            if solution is None or set(solution.keys()) != set(active_vars):
                raise SolverNotValidForRegimeException

            b = sp.ImmutableMatrix([solution[s] for s in active_vars])

            # Solve the system
            try:
                A = (self.coeffs * sp.Symbol(dt_symbol)).exp()
            except NotImplementedError:
                raise SolverNotValidForRegimeException
            C = sp.ImmutableMatrix([A * b]) - b
            x = sp.MatrixSymbol('x_', self.defn.num_time_derivatives, 1)
            update_exprs = A * x + C
            update_exprs = update_exprs.as_explicit()
            # Check for complex values
            if any(len(e.atoms(sp.I)) for e in update_exprs):
                raise SolverNotValidForRegimeException

            # Create lambda functions to evaluate the subsitution
            # of state and parameter variables
            self.updates = self._lambdify_update_exprs(update_exprs, x)
            logger.info("Finished solving linear ODEs for '{}' regime of '{}' "
                        "class".format(self.defn.name, self.parent.defn.name))


class NonlinearRegime(Regime):

    VALID_METHODS = ['rk2', 'rk4', 'euler']

    def __init__(self, defn, parent, integration_method='rk4', **kwargs):  # @UnusedVariable @IgnorePep8
        Regime.__init__(self, defn, parent)
        logger.info("Constructing explicit update equations for '{}' regime of"
                    " '{}' class".format(self.defn.name,
                                         self.parent.defn.name))
        if integration_method not in self.VALID_METHODS:
            raise NineMLUsageError(
                "'{}' is not a valid integration method ('{}')"
                .format(integration_method, "', '".join(self.VALID_METHODS)))
        self.integration_method = integration_method
        # Create vector to represent the state of the regime.
        # NB: time derivatives are sorted by variable name so we can be sure
        # of their order when reinterpreting as state updates.
        x = sp.MatrixSymbol('x_', self.defn.num_time_derivatives, 1)
        # Vectorize time derivative expressions in terms of 'x' state variable
        # vector
        f = sp.Lambda(x, sp.ImmutableMatrix(
            [td.rhs.subs(
                {sv: x[i, 0]
                 for i, sv in enumerate(self.defn.time_derivative_variables)})
             for td in self.defn.time_derivatives]))
        h = sp.Symbol(dt_symbol)  # Step size
        if self.integration_method == 'euler':
            update_exprs = x + h * f(x)
        elif self.integration_method == 'rk2':
            k = h * f
            update_exprs = x + h * f(x + k / 2)
        elif self.integration_method == 'rk4':
            k1 = h * f(x)
            k2 = h * f(x + k1 / 2)
            k3 = h * f(x + k2 / 2)
            k4 = h * f(x + k3)
            update_exprs = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            assert False, "Unrecognised method '{}'"  # Should never get here
        # Create lambda functions to evaluate the subsitution
        # of state and parameter variables
        self.updates = self._lambdify_update_exprs(update_exprs, x)
        logger.info("Finished constructing explicit update equations for '{}' "
                    "regime of '{}' class".format(self.defn.name,
                                                  self.parent.defn.name))


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


class Expression(object):
    """
    A thin wrapper around a lambda function that can be pickled

    Parameters
    ----------
    symbols : list(str)
        The list of symbols to be the substituted
    expr : sympy.Expression
        The Sympy expression
    """

    def __init__(self, symbols, expr):
        self._symbols = symbols
        self._expr = expr
        self.eval = sp.lambdify(symbols, expr)

    def __getinitargs__(self):
        return self._symbols, self._expr
