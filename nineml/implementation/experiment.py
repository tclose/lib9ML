import numpy as np
import os.path as op
from itertools import chain, repeat
from operator import attrgetter
import bisect
import nineml.units as un
from .dynamics import AnalogSendPort, AnalogReceivePort, Port
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class AnalogSource(AnalogSendPort):
    """
    An input source that can be connected to an AnalogReceivePort or
    AnalogSendPort
    """

    def __init__(self, name, signal):
        self._name = name
        self.buffer = [(float(t.in_units(un.s)), float(a.in_si_units()))
                       for t, a in signal]

    @property
    def name(self):
        return self._name

    def connect_to(self, receive_port, delay):
        receive_port.connect_from(self, delay)

    @property
    def _location(self):
        return "anlog source '{}'".format(self.name)

    @classmethod
    def step(cls, name, amplitude, start_t=50 * un.ms,
             stop_t=100 * un.ms, rise_time=0.01 * un.ms):
        off = amplitude.units.dimension.origin
        return AnalogSource(name,
                            [(0 * un.s, off),
                             (start_t - rise_time, off),
                             (start_t, amplitude),
                             (stop_t, amplitude)])


class AnalogSink(AnalogReceivePort):

    DEFAULT_PLOT_STEPS = 100

    def __init__(self, name):
        self._name = name
        self.sender = None
        self.delay = None

    @property
    def name(self):
        return self._name

    @property
    def _location(self):
        return "analog sink '{}'".format(self.name)

    def values(self, times):
        return [
            self.value(float(t.in_units(un.s))
                       if isinstance(t, un.Quantity) else t) for t in times]

    @property
    def dimension(self):
        return self.sender.defn.dimension

    def plot(self, times=None, show=True):
        if plt is None:
            raise ImportError(
                "Cannot plot as matplotlib is not installed")
        if times is None:
            times = self.default_times
        else:
            [(float(t.in_si_units()) if isinstance(t, un.Quantity) else t)
             for t in times]
        fig = plt.figure()
        ax = fig.gca()
        self._plot_trace(ax, times)
        plt.title(self.name)
        plt.ylabel('{} ({})'.format(self.dimension.name,
                                    self.dimension.origin.units.name))
        plt.xlabel('time (s)')
        if show:
            plt.show()
        return fig

    @classmethod
    def save(self, path, sinks, times=None):
        picklable_sinks = []
        for sink in sinks:
            if times is None:
                sink_times = sink.default_times
            else:
                sink_times = times
            picklable_sinks.append(PicklableAnalogSink(
                sink.name, sink_times, sink.values(sink_times),
                sink.dimension))
        with open(path, 'w') as f:
            pkl.dump(picklable_sinks, f)

    @classmethod
    def combined_plot(cls, sinks, times=None, show=True):
        if plt is None:
            raise ImportError(
                "Cannot plot as matplotlib is not installed")
        if times is None:
            times = sinks[0].default_times
        fig = plt.figure()
        ax = fig.gca()
        common_prefix = op.commonprefix([s.name for s in sinks])
        for sink in sinks:
            sink._plot_trace(ax, times, label=sink.name[len(common_prefix):])
        plt.title("{} Signals".format(common_prefix.strip('_')))
        plt.legend()
        dims = set(s.dimension for s in sinks)
        if len(dims) == 1:
            dimension = next(iter(dims))
            plt.ylabel('{} ({})'.format(dimension.name,
                                        dimension.origin.units.name))
        else:
            plt.ylabel('various')
        plt.xlabel('time (s)')
        if show:
            plt.show()
        return fig

    def _plot_trace(self, ax, times, label=None):
        ax.plot(times, self.values(times), label=label)

    @property
    def default_times(self):
        incr = ((self.sender.stop_t - self.sender.start_t) /
                self.DEFAULT_PLOT_STEPS)
        return [t * incr + self.sender.start_t + self.delay
                for t in range(self.DEFAULT_PLOT_STEPS)]


class EventSource(object):

    def __init__(self, name, events):
        self.name = name
        self.events = list(events)

    def connect_to(self, receive_port, delay):
        for event_t in self.events:
            receive_port.receive(float((event_t + delay).in_si_units()))


class EventSink(Port):

    def __init__(self, name):
        self._name = name
        self.events = []

    @property
    def name(self):
        return self._name

    def receive(self, t):
        bisect.insort(self.events, t)

    def plot(self, show=True):
        if plt is None:
            raise ImportError(
                "Cannot plot as matplotlib is not installed")
        fig = plt.figure()
        if self.events:
            plt.scatter(self.events, 0)
        plt.xlabel('Time (ms)')
        plt.title("{} Events".format(self.name))
        if show:
            plt.show()
        return fig

    @classmethod
    def save(self, path, sinks):
        with open(path, 'w') as f:
            pkl.dump(sinks, f)

    @classmethod
    def combined_plot(self, sinks, show=True):
        if plt is None:
            raise ImportError(
                "Cannot plot as matplotlib is not installed")
        sinks = sorted(sinks, key=attrgetter('name'))
        spikes = list(zip(*chain(*(
            zip(s.events, repeat(i)) for i, s in enumerate(sinks)))))
        fig = plt.figure()
        if spikes:
            plt.scatter(*spikes)
        plt.xlabel('Time (ms)')
        plt.ylabel('Cell Indices')
        plt.title("{} Events".format(
            op.commonprefix([s.name for s in sinks]).strip('_')))
        if show:
            plt.show()
        return fig

    @classmethod
    def histogram_plot(cls, sinks, duration, bin_width, show=True):
        if plt is None:
            raise ImportError(
                "Cannot plot as matplotlib is not installed")
        spike_times = np.asarray([s.events for s in sinks])
        plt.figure()
        plt.hist(spike_times, bins=int(np.floor(duration / bin_width)))
        plt.xlabel('Time (ms)')
        plt.ylabel('Rate')
        plt.title("{} PSTH".format(op.commonprefix(s.name for s in sinks)))
        if show:
            plt.show()


class PicklableAnalogSink(AnalogSink):
    """
    A copy of an AnalogSink object that can be pickled to file but then plotted
    as per a regular analog sink
    """

    def __init__(self, name, times, values, dimension):
        self._name = name
        self._times = times
        self._values = values
        self.dimension = dimension

    def values(self, times):
        times = [float(t.in_units(un.s)) if isinstance(t, un.Quantity) else t
                 for t in times]
        assert times == self._times
        return self._values
