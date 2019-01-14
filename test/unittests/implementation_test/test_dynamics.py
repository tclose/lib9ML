from collections import namedtuple
import ninemlcatalog
from collections import OrderedDict
from nineml import units as un
from nineml.user import (DynamicsProperties as DynamicsProperties9ML,
                         MultiDynamicsProperties as MultiDynamicsProperties9ML)

from nineml.implementation.dynamics import (
    Dynamics, AnalogSource, AnalogSink, EventSink, EventSource)
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

    def analog_source(self, name, amplitude, start_t=50 * un.ms,
                      stop_t=100 * un.ms, rise_time=0.01 * un.ms):
        off = amplitude.units.dimension.origin
        return AnalogSource(
            name,
            [(0 * un.s, off),
             (start_t - rise_time, off),
             (start_t, amplitude),
             (stop_t, amplitude)])

    def test_liaf(self, dt=0.001 * un.ms, duration=100.0 * un.ms):

        definition = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                        'PyNNLeakyIntegrateAndFire')
        properties = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                        'PyNNLeakyIntegrateAndFireProperties')
        initial_state = SimpleState(
            {'v': -65.0 * un.mV, 'end_refractory': 0.0 * un.s},
            'subthreshold', definition)
        isyn_in = self.analog_source('isyn', 1 * un.nA)
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
        isyn_in = self.analog_source('isyn_in', 0.02 * un.nA)
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

    def test_izhikevich_fs(self, dt=0.001 * un.ms, duration=100.0 * un.ms):

        definition = ninemlcatalog.load('neuron/Izhikevich',
                                        'IzhikevichFastSpiking')
        properties = ninemlcatalog.load('neuron/Izhikevich',
                                        'SampleIzhikevichFastSpiking')
        initial_state = SimpleState(
            {'V': -65.0 * un.mV, 'U': -1.625 * un.pA}, 'subVb', definition)
        isyn_in = self.analog_source('isyn_in', 100 * un.pA)
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(definition, properties, initial_state,
                            start_t=0.0 * un.s, dt=dt)
        isyn_in.connect_to(dynamics.analog_reduce_ports['iSyn'],
                           delay=0 * un.s)
        dynamics.analog_send_ports['V'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spikeOutput'].connect_to(spike_out,
                                                            0 * un.s)
        dynamics.simulate(duration)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.058, 0.081])
        return v_out

    def test_hodgkin_huxley(self, dt=0.001 * un.ms, duration=100.0 * un.ms):

        definition = ninemlcatalog.load('neuron/HodgkinHuxley',
                                        'PyNNHodgkinHuxley')
        properties = ninemlcatalog.load('neuron/HodgkinHuxley',
                                        'PyNNHodgkinHuxleyProperties')
        initial_state = SimpleState(
            {'v': -65.0 * un.mV, 'm': 0.0 * un.unitless,
             'h': 1.0 * un.unitless, 'n': 0.0 * un.unitless},
            'sole', definition)
        isyn_in = self.analog_source('isyn', 0.5 * un.nA)
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(definition, properties, initial_state,
                            start_t=0.0 * un.s, dt=dt)
        isyn_in.connect_to(dynamics.analog_reduce_ports['iExt'],
                           delay=0 * un.s)
        dynamics.analog_send_ports['v'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['outgoingSpike'].connect_to(spike_out,
                                                              0 * un.s)
        dynamics.simulate(duration)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.038, 0.058, 0.07, 0.082, 0.094])
        return v_out

    def test_liaf_alpha_syn(self, dt=0.001 * un.ms, duration=100.0 * un.ms):
        liaf = ninemlcatalog.load(
            'neuron/LeakyIntegrateAndFire/',
            'PyNNLeakyIntegrateAndFireProperties')
        alpha = ninemlcatalog.load(
            'postsynapticresponse/Alpha', 'SamplePyNNAlphaProperties')
        static = ninemlcatalog.load(
            'plasticity/Static', 'Static')
        static_props = DynamicsProperties9ML(
            'static_props', static, {'weight': 5 * un.nA})
        properties = MultiDynamicsProperties9ML(
            name='IafAlpha_sans_synapses',
            sub_components={
                'cell': liaf,
                'syn': MultiDynamicsProperties9ML(
                    name="IafAlaphSyn",
                    sub_components={'psr': alpha,
                                    'pls': static_props},
                    port_connections=[
                        ('pls', 'fixed_weight', 'psr', 'q')],
                    port_exposures=[('psr', 'i_synaptic'),
                                    ('psr', 'spike')])},
            port_connections=[
                ('syn', 'i_synaptic__psr', 'cell', 'i_synaptic')],
            port_exposures=[('syn', 'spike__psr', 'spike_in'),
                            ('cell', 'v', 'v'),
                            ('cell', 'spike_output', 'spike_out')])
        definition = properties.component_class
        initial_state = SimpleState(
            {'v__cell': -65.0 * un.mV,
             'end_refractory__cell': 0.0 * un.ms,
             'a__psr__syn': 0.0 * un.nA,
             'b__psr__syn': 0.0 * un.nA},
            'subthreshold___sole_____sole', definition)
        spike_in = EventSource('spike_in', [t * un.ms for t in [
            50, 65, 70, 72.5, 71, 73, 72, 70.5, 90]])
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(definition, properties, initial_state,
                            start_t=0.0 * un.s, dt=dt)
        spike_in.connect_to(dynamics.event_receive_ports['spike_in'],
                            delay=0 * un.s)
        dynamics.analog_send_ports['v'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spike_out'].connect_to(spike_out,
                                                          0 * un.s)
        dynamics.simulate(duration)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.071])
        return v_out


    def test_poisson(self, duration=100 * un.ms, dt=0.1, **kwargs):  # @UnusedVariable @IgnorePep8

        definition = ninemlcatalog.load('input/Poisson', 'Poisson')
        properties = DynamicsProperties9ML('PoissonProps',
                                           definition, {'rate': 100 * un.Hz})
        initial_state = SimpleState(
            {'t_next': 0.0 * un.ms}, 'default', definition)
        spike_out = EventSink('spike')
        dynamics = Dynamics(definition, properties, initial_state,
                            start_t=0.0 * un.s, dt=dt)
        dynamics.event_send_ports['spike_output'].connect_to(spike_out,
                                                             0 * un.s)
        dynamics.simulate(duration)
        print(spike_out.events)
#         self.assertEqual([round(t, 3) for t in spike_out.events],
#                          [0.054, 0.057, 0.061, 0.065, 0.068, 0.072, 0.076,
#                           0.08, 0.083, 0.087, 0.091, 0.095, 0.098])


if __name__ == '__main__':
    import time
    import numpy as np
    dt = 0.001 * un.ms
    duration = 100.0 * un.ms
    model = 'hodgkin_huxley'
    start = time.time()
    tester = TestDynamics()
    sink = getattr(tester, 'test_{}'.format(model))(dt=dt, duration=duration)
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
