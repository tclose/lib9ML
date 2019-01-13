from collections import namedtuple
import ninemlcatalog
from collections import OrderedDict
from nineml import units as un
from nineml.implementation.dynamics import (
    Dynamics, AnalogSource, AnalogSink, EventSink)
from unittest import TestCase


class SimpleState(object):
    """
    A placeholder until states are included in 9ML specification
    """

    StateVariable = namedtuple('StateVariable', 'name value')

    def __init__(self, state, regime, component_class):
        self.component_class = component_class
        self.state = OrderedDict((k, float(state[k].in_si_units()))
                                 for k in sorted(state))
        self.regime = regime

    def in_si_units(self):
        return self

    @property
    def variables(self):
        return (self.StateVariable(*i) for i in self.state.items())


class TestDynamics(TestCase):

    def test_liaf(self, dt=0.001 * un.ms, duration=100.0 * un.ms):

        definition = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                        'PyNNLeakyIntegrateAndFire')
        properties = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                        'PyNNLeakyIntegrateAndFireProperties')
        initial_state = SimpleState(
            {'v': -65.0 * un.mV, 'end_refractory': 0.0 * un.s},
            'subthreshold', definition)
        i_synaptic_in = AnalogSource(
            'i_synaptic_in',
            [(0 * un.ms, 0 * un.nA),
             (49.99 * un.ms, 0 * un.nA),
             (50 * un.ms, 1 * un.nA),
             (100 * un.ms, 1 * un.nA)])
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(definition, properties, initial_state,
                            start_t=0.0 * un.s, dt=dt)
        i_synaptic_in.connect_to(dynamics.analog_reduce_ports['i_synaptic'],
                                 delay=0 * un.s)
        dynamics.analog_send_ports['v'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spike_output'].connect_to(spike_out,
                                                             0 * un.s)
        dynamics.simulate(duration)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.055, 0.061, 0.068, 0.075, 0.081, 0.088, 0.095])
        return v_out


# import time
# start = time.time()
# sink = TestDynamics().test_liaf()
# end = time.time()
# elapsed = end - start
# print("Simulated {} at {} resolution in {} seconds".format(duration, dt,
#                                                            elapsed))
# sink.plot([t * un.ms for t in np.arange(0, 100, 0.1)])
# print("Plotted results")
