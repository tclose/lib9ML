from collections import namedtuple
import ninemlcatalog
from collections import OrderedDict
from nineml import units as un
from nineml.implementation.dynamics import (
    Dynamics, AnalogSource, AnalogSink, EventSink)
from nineml.exceptions import NineMLUsageError
if __name__ == '__main__':
    class TestCase(object):

        def assertEqual(self, *args, **kwargs):
            pass

else:
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
        isyn_in = AnalogSource(
            'isyn_in',
            [(0 * un.ms, 0 * un.nA),
             (49.99 * un.ms, 0 * un.nA),
             (50 * un.ms, 1 * un.nA),
             (100 * un.ms, 1 * un.nA)])
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(definition, properties, initial_state,
                            start_t=0.0 * un.s, dt=dt)
        isyn_in.connect_to(dynamics.analog_reduce_ports['i_synaptic'],
                                 delay=0 * un.s)
        dynamics.analog_send_ports['v'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spike_output'].connect_to(spike_out,
                                                             0 * un.s)
        dynamics.simulate(duration)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.055, 0.061, 0.068, 0.075, 0.081, 0.088, 0.095])
        return v_out

    def test_izhikevich(self, dt=0.001 * un.ms, duration=100.0 * un.ms):

        definition = ninemlcatalog.load('neuron/Izhikevich', 'Izhikevich')
        properties = ninemlcatalog.load('neuron/Izhikevich',
                                        'SampleIzhikevich')
        initial_state = SimpleState(
            {'V': -65.0 * un.mV, 'U': -14.0 * un.mV / un.ms},
            'subthreshold_regime', definition)
        isyn_in = AnalogSource(
            'isyn_in',
            [(0 * un.ms, 0 * un.nA),
             (49.99 * un.ms, 0 * un.nA),
             (50 * un.ms, 0.02 * un.nA),
             (100 * un.ms, 0.02 * un.nA)])
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(definition, properties, initial_state,
                            start_t=0.0 * un.s, dt=dt)
        isyn_in.connect_to(dynamics.analog_reduce_ports['Isyn'],
                           delay=0 * un.s)
        dynamics.analog_send_ports['V'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spike'].connect_to(spike_out,
                                                             0 * un.s)
        dynamics.simulate(duration)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.054, 0.057, 0.061, 0.065, 0.068, 0.072, 0.076,
                          0.08, 0.083, 0.087, 0.091, 0.095, 0.098])
        return v_out


if __name__ == '__main__':
    import time
    import numpy as np
    dt = 0.001 * un.ms
    duration = 100.0 * un.ms
    model = 'izhikevich'
    start = time.time()
    tester = TestDynamics()
    if model == 'liaf':
        sink = tester.test_liaf(dt=dt, duration=duration)
    elif model == 'izhikevich':
        sink = tester.test_izhikevich(dt=dt, duration=duration)
    else:
        assert False, "Unrecognised model '{}'".format(model)
    end = time.time()
    elapsed = end - start
    print("Simulated {} model for {} with {} resolution in {} (real-world) "
          "seconds".format(model, duration, dt, elapsed))
    try:
        sink.plot([t * un.ms for t in np.arange(0, duration.in_units(un.ms),
                                                0.1)])
        print("Plotted results")
    except ImportError:
        print("Could not plot results as matplotlib is not installed")
