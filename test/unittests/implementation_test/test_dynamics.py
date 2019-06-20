import os
from future.utils import PY2
import ninemlcatalog
import numpy.random
from nineml import units as un
from nineml.user import DynamicsProperties, MultiDynamicsProperties
from nineml.implementation import (
    Dynamics, AnalogSource, AnalogSink, EventSink, EventSource)
from nineml.utils.testing.rng import random_state
if __name__ == '__main__':

    class TestCase(object):

        def assertEqual(self, *args, **kwargs):
            pass

    def skip(reason):
        """Dummy skipIf that just returns original function"""
        def decorator(test):  # @UnusedVariable
            def error_message(*args, **kwargs):
                raise Exception(reason)
        return decorator

    def skipIf(condition, reason):
        """Dummy skipIf that just returns original function"""
        def decorator(test):
            if condition:
                def error_message(*args, **kwargs):
                    raise Exception(reason)
            else:
                # Else return plain test function
                return test
        return decorator
else:
    from unittest import TestCase, skipIf

try:
    DISABLE_SIM_TESTS = 'NINEML_DISABLE_SIM_TESTS' in os.environ
except KeyError:
    DISABLE_SIM_TESTS = False


class TestDynamics(TestCase):

    @skipIf(DISABLE_SIM_TESTS, "Simulation tests disabled")
    def test_liaf(self, dt=0.001 * un.ms, duration=100.0 * un.ms,
                  show_progress=False):

        properties = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                        'PyNNLeakyIntegrateAndFireProperties')
        initial_state = {'v': -65.0 * un.mV, 'end_refractory': 0.0 * un.s}
        initial_regime = 'subthreshold'
        isyn_in = AnalogSource.step('isyn', 1 * un.nA)
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(properties,
                            random_state=random_state,
                            initial_state=initial_state,
                            initial_regime=initial_regime,
                            start_t=0.0 * un.s)
        isyn_in.connect_to(dynamics.analog_reduce_ports['i_synaptic'],
                                 delay=0 * un.s)
        dynamics.analog_send_ports['v'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spike_output'].connect_to(spike_out,
                                                             0 * un.s)
        dynamics.simulate(duration, dt=dt, show_progress=show_progress)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.055, 0.061, 0.068, 0.075, 0.081, 0.088, 0.095])
        return v_out

    @skipIf(DISABLE_SIM_TESTS, "Simulation tests disabled")
    def test_izhikevich(self, dt=0.001 * un.ms, duration=100.0 * un.ms,
                        show_progress=False):

        properties = ninemlcatalog.load('neuron/Izhikevich',
                                        'SampleIzhikevich')
        initial_state = {'V': -65.0 * un.mV, 'U': -14.0 * un.mV / un.ms}
        initial_regime = 'subthreshold_regime'
        isyn_in = AnalogSource.step('isyn_in', 0.02 * un.nA)
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(properties,
                            random_state=random_state,
                            initial_state=initial_state,
                            initial_regime=initial_regime,
                            start_t=0.0 * un.s)
        isyn_in.connect_to(dynamics.analog_reduce_ports['Isyn'],
                           delay=0 * un.s)
        dynamics.analog_send_ports['V'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spike'].connect_to(spike_out,
                                                             0 * un.s)
        dynamics.simulate(duration, dt=dt, show_progress=show_progress)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.054, 0.057, 0.061, 0.065, 0.068, 0.072, 0.076,
                          0.08, 0.083, 0.087, 0.091, 0.095, 0.098])
        return v_out

    @skipIf(DISABLE_SIM_TESTS, "Simulation tests disabled")
    def test_izhikevich_fs(self, dt=0.001 * un.ms, duration=100.0 * un.ms,
                           show_progress=False):

        properties = ninemlcatalog.load('neuron/Izhikevich',
                                        'SampleIzhikevichFastSpiking')
        initial_state = {'V': -65.0 * un.mV, 'U': -1.625 * un.pA}
        initial_regime = 'subVb'
        isyn_in = AnalogSource.step('isyn_in', 100 * un.pA)
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(properties,
                            random_state=random_state,
                            initial_state=initial_state,
                            initial_regime=initial_regime,
                            start_t=0.0 * un.s)
        isyn_in.connect_to(dynamics.analog_reduce_ports['iSyn'],
                           delay=0 * un.s)
        dynamics.analog_send_ports['V'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spikeOutput'].connect_to(spike_out,
                                                            0 * un.s)
        dynamics.simulate(duration, dt=dt, show_progress=show_progress)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.058, 0.081])
        return v_out

    @skipIf(DISABLE_SIM_TESTS or PY2,
            "Simulation tests disabled" if DISABLE_SIM_TESTS else
            "Generated equations for HodgkinHuxley overflow on Python 2")
    def test_hodgkin_huxley(self, dt=0.001 * un.ms, duration=100.0 * un.ms,
                            show_progress=False):

        properties = ninemlcatalog.load('neuron/HodgkinHuxley',
                                        'PyNNHodgkinHuxleyProperties')
        initial_state = {'v': -65.0 * un.mV, 'm': 0.0 * un.unitless,
                         'h': 1.0 * un.unitless, 'n': 0.0 * un.unitless}
        initial_regime = 'sole'
        isyn_in = AnalogSource.step('isyn', 0.5 * un.nA)
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(properties,
                            random_state=random_state,
                            initial_state=initial_state,
                            initial_regime=initial_regime,
                            start_t=0.0 * un.s)
        isyn_in.connect_to(dynamics.analog_reduce_ports['iExt'],
                           delay=0 * un.s)
        dynamics.analog_send_ports['v'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['outgoingSpike'].connect_to(spike_out,
                                                              0 * un.s)
        dynamics.simulate(duration, dt=dt, show_progress=show_progress)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.038, 0.058, 0.07, 0.082, 0.094])
        return v_out

    @skipIf(DISABLE_SIM_TESTS, "Simulation tests disabled")
    def test_liaf_alpha_syn(self, dt=0.001 * un.ms, duration=100.0 * un.ms,
                            show_progress=False):
        liaf = ninemlcatalog.load(
            'neuron/LeakyIntegrateAndFire/',
            'PyNNLeakyIntegrateAndFireProperties')
        alpha = ninemlcatalog.load(
            'postsynapticresponse/Alpha', 'SamplePyNNAlphaProperties')
        static = ninemlcatalog.load(
            'plasticity/Static', 'Static')
        static_props = DynamicsProperties(
            'static_props', static, {'weight': 5 * un.nA})
        properties = MultiDynamicsProperties(
            name='IafAlpha_with_alpha_syn',
            sub_components={
                'cell': liaf,
                'syn': MultiDynamicsProperties(
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
        initial_state = {'v__cell': -65.0 * un.mV,
                         'end_refractory__cell': 0.0 * un.ms,
                         'a__psr__syn': 0.0 * un.nA,
                         'b__psr__syn': 0.0 * un.nA}
        initial_regime = 'subthreshold___sole_____sole'
        spike_in = EventSource('spike_in', [t * un.ms for t in [
            50, 65, 70, 72.5, 71, 73, 72, 70.5, 90]])
        v_out = AnalogSink('v')
        spike_out = EventSink('spike')
        dynamics = Dynamics(properties,
                            random_state=random_state,
                            initial_state=initial_state,
                            initial_regime=initial_regime,
                            start_t=0.0 * un.s)
        spike_in.connect_to(dynamics.event_receive_ports['spike_in'],
                            delay=0 * un.s)
        dynamics.analog_send_ports['v'].connect_to(v_out, 0 * un.s)
        dynamics.event_send_ports['spike_out'].connect_to(spike_out,
                                                          0 * un.s)
        dynamics.simulate(duration, dt=dt, show_progress=show_progress)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.071])
        return v_out

    @skipIf(DISABLE_SIM_TESTS, "Simulation tests disabled")
    def test_poisson(self, duration=100 * un.ms, dt=0.1 * un.ms,
                     show_progress=False, **kwargs):  # @UnusedVariable @IgnorePep8

        definition = ninemlcatalog.load('input/Poisson', 'Poisson')
        properties = DynamicsProperties('PoissonProps', definition,
                                        {'rate': 100 * un.Hz})
        initial_state = {'t_next': 0.0 * un.ms}
        initial_regime = 'default'
        spike_out = EventSink('spike')
        dynamics = Dynamics(properties,
                            random_state=random_state,
                            initial_state=initial_state,
                            initial_regime=initial_regime,
                            start_t=0.0 * un.s)
        dynamics.event_send_ports['spike_output'].connect_to(spike_out,
                                                             0 * un.s)
        # Set to fixed seed
        numpy.random.seed(12345)
        dynamics.simulate(duration, dt=dt, show_progress=show_progress)
        self.assertEqual([round(t, 3) for t in spike_out.events],
                         [0.0, 0.027, 0.03, 0.032, 0.035, 0.043, 0.052, 0.085,
                          0.096])
        return spike_out


if __name__ == '__main__':
    import numpy as np
    dt = 0.001 * un.ms
    duration = 100.0 * un.ms
    model = 'liaf'
    tester = TestDynamics()
    sink = getattr(tester, 'test_{}'.format(model))(dt=dt, duration=duration)
    if isinstance(sink, AnalogSink):
        try:
            sink.plot([t * un.ms
                       for t in np.arange(0, duration.in_units(un.ms), 0.1)])
            print("Plotted results")
        except ImportError:
            print("Could not plot results as matplotlib is not installed")
    else:
        print([round(t, 3) for t in sink.events])
