from collections import defaultdict
import ninemlcatalog
import math
import random
from nineml import units as un
from nineml.user import Property as Property
from nineml.implementation import Network, EventSink, AnalogSink
from nineml.exceptions import NineMLNameError


if __name__ == '__main__':

    class TestCase(object):

        def assertEqual(self, *args, **kwargs):
            pass

    def skip(reason):
        """Dummy skip that just returns original function"""
        def decorator(test):  # @UnusedVariable
            return test
        return decorator


else:
    from unittest import TestCase, skip

LARGE_INT = 2 ** 31 - 1


class TestNetwork(TestCase):

    @skip("Network test isn't ready yet")
    def test_brunel(self, case='AI', order=50, duration=250.0 * un.ms,
                    dt=0.01 * un.ms, random_seed=None):
        random.seed(random_seed)
        model = self._reduced_brunel_9ml(case, order, random_seed=random_seed)
        network = Network(model, start_t=0 * un.s)
        event_sinks = defaultdict(list)
        analog_sinks = defaultdict(list)
        for comp in network.components:
            event_sink = EventSink(comp.name + '_spike_sink')
            analog_sink = AnalogSink(comp.name + '_v_sink')
            if comp.name.startswith('Exc'):
                comp.ports['spike_output__LeakyIntegrateAndFire'].connect_to(
                    event_sink, network.min_delay)
                comp.ports['v__LeakyIntegrateAndFire'].connect_to(
                    analog_sink, network.min_delay)
                event_sinks['Exc'].append(event_sink)
                analog_sinks['Exc'].append(analog_sink)
            elif comp.name.startswith('Inh'):
                comp.ports['spike_output__LeakyIntegrateAndFire'].connect_to(
                    event_sink, network.min_delay)
                event_sinks['Inh'].append(event_sink)
                comp.ports['v__LeakyIntegrateAndFire'].connect_to(
                    analog_sink, network.min_delay)
                event_sinks['Inh'].append(event_sink)
                analog_sinks['Inh'].append(analog_sink)
            elif comp.name.startswith('Ext'):
                comp.ports['spike_output'].connect_to(event_sink,
                                                      network.min_delay)
                event_sinks['Ext'].append(event_sink)
            else:
                assert False, "Unrecognised component '{}'".format(comp.name)
        network.simulate(duration, dt=dt)
        return event_sinks, analog_sinks

    def _reduced_brunel_9ml(self, case, order, random_seed=None):

        model = ninemlcatalog.load('network/Brunel2000/' + case).as_network(
            'Brunel_{}'.format(case))
        model = model.clone()
        if random_seed is not None:
            for projection in model.projections:
                projection.connectivity.seed = random.randint(0, LARGE_INT)
        scale = order / model.population('Inh').size
        # rescale populations
        for pop in model.populations:
            pop.size = int(math.ceil(pop.size * scale))
        for proj in model.projections:
            props = proj.connectivity.rule_properties
            proj.connectivity._source_size = proj.pre.size
            proj.connectivity._destination_size = proj.post.size
            try:
                number = props.property('number')
                props.set(Property(
                    number.name,
                    int(math.ceil(float(number.value) * scale)) * un.unitless))
            except NineMLNameError:
                pass

        return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dt = 0.01 * un.ms
    case = 'AI'
    order = 10
    duration = 20.0 * un.ms
    model = 'brunel'
    print("Simulating {} model for {} with {} resolution".format(
        model, duration, dt))
    tester = TestNetwork()
    test = getattr(tester, 'test_{}'.format(model))
    event_sinks, analog_sinks = test(dt=dt, duration=duration, case=case,
                                     order=order, random_seed=12345)
    for pop_sinks in event_sinks.values():
        EventSink.combined_plot(pop_sinks, show=False)
    for pop_sinks in analog_sinks.values():
        AnalogSink.combined_plot(pop_sinks[:10], show=False)
    plt.show()
