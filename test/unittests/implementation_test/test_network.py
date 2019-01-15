import ninemlcatalog
import math
from nineml import units as un
from nineml.user import Property as Property9ML
from nineml.implementation import Network, EventSink
if __name__ == '__main__':
    class TestCase(object):

        def assertEqual(self, *args, **kwargs):
            pass

else:
    from unittest import TestCase


class TestNetwork(TestCase):

    def test_brunel(self, case='AI', order=50, duration=250.0 * un.ms,
                    dt=0.01 * un.ms):
        model = self._reduced_brunel_9ml(case, order)
        network = Network(model, initial_states, start_t=0 * un.s,
                          dt=dt)
        event_sinks = {}
        network.simulate(duration)
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
        for proj in (model.projection('Excitation'),
                     model.projection('Inhibition')):
            props = proj.connectivity.rule_properties
            number = props.property('number')
            props.set(Property9ML(
                number.name,
                int(math.ceil(float(number.value) * scale)) * un.unitless))
        return model


if __name__ == '__main__':
    import time
    dt = 0.01 * un.ms
    case = 'AI'
    order = 10
    duration = 20.0 * un.ms
    model = 'brunel'
    start = time.time()
    tester = TestNetwork()
    sinks = getattr(tester, 'test_{}'.format(model))(dt=dt, duration=duration,
                                                     case=case, order=order)
    end = time.time()
    elapsed = end - start
    print("Simulated {} model for {} with {} resolution in {} (real-world) "
          "seconds".format(model, duration, dt, elapsed))
    for pop_sinks in sinks.values():
        EventSink.combined_plot(pop_sinks)
