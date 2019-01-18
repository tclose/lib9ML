import ninemlcatalog
import math
from nineml import units as un
from nineml.user import Property as Property
from nineml.implementation import Network, EventSink
import unittest
from nineml.exceptions import NineMLNameError
if __name__ == '__main__':
    class TestCase(object):

        def assertEqual(self, *args, **kwargs):
            pass

else:
    from unittest import TestCase


class TestNetwork(TestCase):

#     @unittest.skip
    def test_brunel(self, case='AI', order=50, duration=250.0 * un.ms,
                    dt=0.01 * un.ms):
        model = self._reduced_brunel_9ml(case, order)
        network = Network(model, start_t=0 * un.s)
        event_sinks = {}
        network.simulate(duration, dt=dt)
        self.assertEqual([len(s) for s in event_sinks],
                         [])
        return event_sinks

    def _reduced_brunel_9ml(self, case, order):

        model = ninemlcatalog.load('network/Brunel2000/' + case).as_network(
            'Brunel_{}'.format(case))
        model = model.clone()
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
    dt = 0.01 * un.ms
    case = 'AI'
    order = 10
    duration = 20.0 * un.ms
    model = 'brunel'
    print("Simulating {} model for {} with {} resolution".format(
        model, duration, dt))
    tester = TestNetwork()
    sinks = getattr(tester, 'test_{}'.format(model))(dt=dt, duration=duration,
                                                     case=case, order=order)
    for pop_sinks in sinks.values():
        EventSink.combined_plot(pop_sinks)
