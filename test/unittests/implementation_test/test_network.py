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
                    dt=0.01 * un.ms, random_seed=None, num_processes=1,
                    **kwargs):
        random.seed(random_seed)
        model = ninemlcatalog.load('network/Brunel2000/' + case).as_network(
            'Brunel_{}'.format(case))
        if order is not None:
            model = self._reduced_brunel(model, order, random_seed=random_seed)
        network = Network(model, start_t=0 * un.s, num_processes=num_processes,
                          sinks=[('Exc__cell', 'spike_output', range(250)),
                                 ('Inh__cell', 'spike_output', range(250)),
#                                  ('Ext__cell', 'spike_output', range(250)),
#                                  ('Exc__cell', 'v', range(10)),
#                                  ('Inh__cell', 'v', range(10))
                                 ])
        network.simulate(duration, dt=dt, **kwargs)
        # Detach sinks and return
        return {name: [s.detach() for s in sink_group]
                for name, sink_group in network.sinks.items()}

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
    import errno
    import sys
    from argparse import ArgumentParser
    import logging
    import pickle as pkl
    # Disable tqdm locking, which causes issues with PyPy
    from tqdm import tqdm
    tqdm.get_lock().locks = []

    logger = logging.getLogger('nineml')

    parser = ArgumentParser()
    parser.add_argument('--dt', default=0.01, type=float,
                        help="time step size")
    parser.add_argument('--case', default='AI', help="Brunel network case")
    parser.add_argument('--order', default=50, type=int,
                        help=("The number of inhibitory neurons. The rest of "
                              "the network is scaled accordingly "
                              "(original=1000)"))
    parser.add_argument('--duration', default=50.0, type=float,
                        help="The duration of the simulation (ms)")
    parser.add_argument('--save_figs', default=None, type=str, metavar='PATH',
                        help=("The location of the directory to save the "
                              "generated figures"))
    parser.add_argument('--save_sinks', default=None, type=str,
                        help=("Save sinks instead of plotting results"))
    parser.add_argument('--load_sinks', default=None, type=str,
                        help=("Load previously saved sinks and plot"))
    parser.add_argument('--hide_progress', default=False, action='store_true',
                        help="Whether to hide progress bar")
    parser.add_argument('--num_processes', default=1, type=int,
                        help="Number of processes to use")
    parser.add_argument('--loglevel', default='warning', type=str,
                        help=("The level at which to display logging. Can be "
                              "one of 'debug', 'info', warning or 'error'"))
    parser.add_argument('--nplot', default=False, type=int,
                        help=("The number of neurons to plot"))
    args = parser.parse_args()

    if args.loglevel.lower() == 'debug':
        loglevel = logging.DEBUG
    elif args.loglevel.lower() == 'info':
        loglevel = logging.INFO
    elif args.loglevel.lower() == 'warning':
        loglevel = logging.WARNING
    elif args.loglevel.lower() == 'error':
        loglevel = logging.ERROR
    else:
        raise Exception("Unrecognised log-level '{}'".format(args.loglevel))

    logger.setLevel(loglevel)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.save_figs:
        try:
            os.makedirs(args.save_figs)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    if args.save_sinks and args.load_sinks:
        raise Exception("Why load and save sinks in the same command?")

    if args.load_sinks:
        with open(args.load_sinks, 'rb') as f:
            sinks = pkl.load(f)
    else:
        dt = args.dt * un.ms
        duration = args.duration * un.ms
        num_processes = args.num_processes if args.num_processes > 0 else None
        model = 'brunel'
        tester = TestNetwork()
        test = getattr(tester, 'test_{}'.format(model))
        sinks = test(dt=dt, duration=duration, case=args.case,
                     order=args.order, random_seed=12345,
                     show_progress=(not args.hide_progress),
                     num_processes=num_processes)
    if args.save_sinks:
        with open(args.save_sinks, 'wb') as f:
            pkl.dump(sinks, f)
        logger.info("Saved sinks to '{}'".format(args.save_sinks))
    else:
        for pop_sinks in sinks.values():
            if args.nplot:
                pop_sinks = pop_sinks[:args.nplot]
            fig = pop_sinks[0].combined_plot(pop_sinks, show=False)
            if args.save_figs:
                common_prefix = op.commonprefix([s.name for s in pop_sinks])
                filename = common_prefix + '.png'
                fig.set_size_inches(10, 10)
                fig_path = op.join(args.save_figs, filename)
                plt.savefig(fig_path)
                logger.info("Saved '{}' figure to '{}'".format(
                    common_prefix, fig_path))
        if not args.save_figs:
            plt.show()
