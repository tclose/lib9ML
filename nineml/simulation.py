
class Dynamics(object):

    def __init__(self, properties):
        self.properties = properties


class Regime(object):

    def __init__(self, definition):
        self.definition = definition

    def simulate(self, start_t, stop_t, state, solver):
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
        t = start_t
        while t < stop_t:
            new_state, new_t = solver.step(self, state, t)
            for oc in self.on_conditions:
                oc.check(state, new_state, t, new_t)
            state = new_state
            t = new_t
        return state, t


class OnCondition(object):

    def __init__(self, definition):
        self.definition = definition


class OnEvent(object):

    def __init__(self, definition):
        self.definition = definition


class Network(object):

    def __init__(self, populations, projections):
        self.populations = populations
        self.projections = projections


class Event(Exception):

    def __init__(self, name, time, exact_state, cell_id=None):
        self.name = name
        self.time = time
        self.state = exact_state
        self.cell_id = cell_id
