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
        network = Network(model, start_t=0 * un.s,
                          sinks=[('Exc__cell', 'spike_output'),
                                 ('Inh__cell', 'spike_output'),
                                 ('Ext__cell', 'spike_output'),
                                 ('Exc__cell', 'v', range(10)),
                                 ('Inh__cell', 'v')])
        network.simulate(duration, dt=dt)
        return network.sinks

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
    tester = TestNetwork()
    test = getattr(tester, 'test_{}'.format(model))
    sinks = test(dt=dt, duration=duration, case=case, order=order,
                 random_seed=12345)
    for pop_sinks in sinks.values():
        pop_sinks[0].combined_plot(pop_sinks, show=False)
    plt.show()
