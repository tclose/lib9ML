from collections import namedtuple
import time
import ninemlcatalog
from collections import OrderedDict
import numpy as np
from nineml import units as un
from nineml.implementation.dynamics import Dynamics, AnalogSource, AnalogSink
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
# from unittest import TestCase


class SimpleState(object):

    StateVariable = namedtuple('StateVariable', 'name value')

    def __init__(self, state, regime, component_class):
        self.component_class = component_class
        self.state = OrderedDict((k, state[k]) for k in sorted(state))
        self.regime = regime

    def in_si_units(self):
        return self

    @property
    def variables(self):
        return (self.StateVariable(*i) for i in self.state.items())


# class TestDynamics(TestCase):

#     def test_liaf(self, dt=0.001 * un.ms, duration=100.0 * un.ms):

def test_simulation(dt, duration):

    definition = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                    'PyNNLeakyIntegrateAndFire')
    properties = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                    'PyNNLeakyIntegrateAndFireProperties')
    initial_state = SimpleState(
        {'v': -65.0 * un.mV, 'end_refractory': 0.0 * un.s},
        'subthreshold', definition)
    input_signal = AnalogSource(
        'step', [(0 * un.ms, 0 * un.nA),
                 (49.99 * un.ms, 0 * un.nA),
                 (50 * un.ms, 1 * un.nA),
                 (100 * un.ms, 1 * un.nA)])
    recorder = AnalogSink('v')
    dynamics = Dynamics(definition, properties, initial_state,
                        start_t=0.0 * un.s, dt=dt)
    input_signal.connect_to(dynamics.analog_reduce_ports['i_synaptic'],
                            delay=0 * un.s)
    dynamics.analog_send_ports['v'].connect_to(recorder, 0 * un.s)
    dynamics.simulate(duration)
    return recorder


def plot(recorder, times):
    if plt is None:
        raise Exception("Cannot plot as matplotlib is not installed")
    plt.figure()
    plt.plot(times, recorder.values(times))
    plt.title(recorder.name + ' ({})'.format(
        recorder.dimension.origin.units.name))
    plt.show()


dt = 0.001 * un.ms
duration = 100.0 * un.ms

start = time.time()
recorder = test_simulation(dt, duration)
end = time.time()
elapsed = end - start
print("Simulated {} at {} resolution in {} seconds".format(duration, dt,
                                                           elapsed))
plot(recorder, [t * un.ms for t in np.arange(0, 100, 0.1)])
