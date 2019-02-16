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
        model = ninemlcatalog.load('network/Brunel2000/' + case).as_network(
            'Brunel_{}'.format(case))
        if order is not None:
            model = self._reduced_brunel(model, order, random_seed=random_seed)
        network = Network(model, start_t=0 * un.s,
                          sinks=[('Exc__cell', 'spike_output', range(250)),
                                 ('Inh__cell', 'spike_output', range(250)),
                                 ('Ext__cell', 'spike_output', range(250)),
                                 ('Exc__cell', 'v', range(10)),
                                 ('Inh__cell', 'v', range(10))])
        network.simulate(duration, dt=dt)
        return network.sinks

    def _reduced_brunel(self, model, order, random_seed=None):
        model = model.clone()
        if random_seed is not None:
            for projection in model.projections:
                projection.connectivity.seed = random.randint(0, LARGE_INT)
        scale = float(order) / model.population('Inh').size
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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    import os.path as op
    import os
    import sys
    from argparse import ArgumentParser
    import logging
    import pickle as pkl

    logger = logging.getLogger('nineml')

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = ArgumentParser()
    parser.add_argument('--dt', default=0.01, type=float,
                        help="time step size")
    parser.add_argument('--case', default='AI', help="Brunel network case")
    parser.add_argument('--order', default=50, type=int,
                        help=("The number of inhibitory neurons. The rest of "
                              "the network is scaled accordingly "
                              "(original=1000)"))
    parser.add_argument('--duration', default=50.0, type=float,
                        help="The duration of the simulation")
    parser.add_argument('--save_figs', default=None, type=str, metavar='PATH',
                        help=("The location of the directory to save the "
                              "generated figures"))
    parser.add_argument('--save_sinks', default=None, type=str,
                        help=("Save sinks instead of plotting results"))
    parser.add_argument('--load_sinks', default=None, type=str,
                        help=("Load previously saved sinks and plot"))
    args = parser.parse_args()

    if args.save_figs:
        os.makedirs(args.save_figs, exist_ok=True)

    if args.save_sinks and args.load_sinks:
        raise Exception("Can't load and save sinks simultaneously")

    if not args.load_sinks:
        dt = args.dt * un.ms
        duration = args.duration * un.ms
        model = 'brunel'
        tester = TestNetwork()
        test = getattr(tester, 'test_{}'.format(model))
        sinks = test(dt=dt, duration=duration, case=args.case,
                     order=args.order, random_seed=12345)
    else:
        with open(args.load_sinks) as f:
            sinks = pkl.load(f)
    if args.save_sinks is None:
        for pop_sinks in sinks.values():
            fig = pop_sinks[0].combined_plot(pop_sinks, show=False)
            if args.save_figs:
                filename = (op.commonprefix([s.name for s in pop_sinks]) +
                            '.png')
                fig.set_size_inches(10, 10)
                plt.savefig(op.join(args.save_figs, filename))
        if not args.save_figs:
            plt.show()
    else:
        with open(args.save_sinks, 'wb') as f:
            sinks = {
                pop_name: [s.picklable() for s in pop_sinks]
                for pop_name, pop_sinks in sinks.items()}
            pkl.dump(sinks, f)
