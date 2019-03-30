import os
import ninemlcatalog
import math
import random
import logging
from itertools import chain
from nineml import units as un
from nineml.user import Property as Property
from nineml.implementation import Network
from nineml.exceptions import NineMLNameError

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


logger = logging.getLogger('nineml')


if __name__ == '__main__':

    class TestCase(object):

        def assertEqual(self, *args, **kwargs):
            pass

    def skip(reason):
        """Dummy skip that just returns original function"""
        def decorator(test):  # @UnusedVariable
            return test
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

LARGE_INT = 2 ** 31 - 1

try:
    DISABLE_SIM_TESTS = os.environ['DISABLE_SIM_TESTS']
except KeyError:
    DISABLE_SIM_TESTS = False


class TestNetwork(TestCase):

    ref_rate = {'Exc__cell_spike_output': (11612.0 * un.per_ms, 0.075),
                'Inh__cell_spike_output': (11604.0 * un.per_ms, 0.075),
                'Ext__cell_spike_output': (2622584.0 * un.per_ms, 0.075)}

    @skipIf(DISABLE_SIM_TESTS, "Simulation tests have been disabled")
    def test_brunel(self, case='AI', order=50, duration=250.0 * un.ms,
                    dt=0.01 * un.ms, random_seed=None, num_processes=4,
                    nrecord=50, record_v=False, nrecord_v=5, record_ext=True,
                    **kwargs):
        random.seed(random_seed)
        model = ninemlcatalog.load('network/Brunel2000/' + case).as_network(
            'Brunel_{}'.format(case))
        if order is not None:
            model = self._reduced_brunel(model, order, random_seed=random_seed)
        sink_specs = [('Exc__cell', 'spike_output', range(nrecord)),
                      ('Inh__cell', 'spike_output', range(nrecord))]
        if record_v:
            sink_specs.append(('Exc__cell', 'v', range(nrecord_v)))
        if record_ext:
            sink_specs.append(('Ext__cell', 'spike_output', range(nrecord)))
        network = Network(model, start_t=0 * un.s, num_processes=num_processes,
                          sinks=sink_specs)
        network.simulate(duration, dt=dt, **kwargs)
        sinks = {name: [s.detach() for s in sink_group]
                 for name, sink_group in network.sinks.items()}
        event_sink_names = ['Exc__cell_spike_output', 'Inh__cell_spike_output']
        if record_ext:
            event_sink_names.append('Ext__cell_spike_output')
#        for sink_name in event_sink_names:
#            spike_times = list(chain(*(s.events for s in sinks[sink_name])))
#            rate = len(spike_times) / (duration * nrecord)
#            ref_rate, rate_tol = self.ref_rate[sink_name]
#            rate_error = (abs((rate - ref_rate).in_si_units()) /
#                          ref_rate.in_si_units())
#            self.assertLess(rate_error, rate_tol,
#                            "Spike rate of '{}' ({}) is not within the "
#                            "given tolerance ({}%) of the reference ({})"
#                            .format(sink_name, rate, 100 * rate_tol,
#                                    ref_rate))
        # Detach sinks and return
        return sinks

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
    import errno
    import sys
    from argparse import ArgumentParser
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
    parser.add_argument('--record_ext', action='store_true', default=False,
                        help=("Whether to record the external stimulation "
                              "events or not"))
    parser.add_argument('--record_v', action='store_true', default=False,
                        help=("Whether to record the voltage traces of the "
                              "excitatory cell or not"))
    parser.add_argument('--nrecord', default=False, type=int,
                        help=("The number of neurons to record spikes from"))
    parser.add_argument('--nrecord_v', default=False, type=int,
                        help=("The number of neurons to record v from"))
    parser.add_argument('--logfile', default=None, type=str,
                        help="The path of a file to write the log to")
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
    if args.logfile is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(args.logfile)
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
                     num_processes=num_processes, nrecord=args.nrecord,
                     record_v=args.record_v, nrecord_v=args.nrecord_v,
                     record_ext=args.record_ext)
    if args.save_sinks:
        with open(args.save_sinks, 'wb') as f:
            pkl.dump(sinks, f)
        logger.info("Saved sinks to '{}'".format(args.save_sinks))
    else:
        for pop_sinks in sinks.values():
            common_prefix = op.commonprefix([s.name for s in pop_sinks])
            fig = pop_sinks[0].combined_plot(pop_sinks, show=False)
            print("plotted {}".format(common_prefix))
            if args.save_figs:
                filename = common_prefix + '.png'
                fig.set_size_inches(10, 10)
                fig_path = op.join(args.save_figs, filename)
                plt.savefig(fig_path)
                logger.info("Saved '{}' figure to '{}'".format(
                    common_prefix, fig_path))
        if not args.save_figs:
            plt.show()
