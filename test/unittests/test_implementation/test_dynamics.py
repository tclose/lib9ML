#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import ninemlcatalog
from collections import namedtuple
from nineml import units as un
from nineml.implementation.dynamics import Dynamics, AnalogSource
# from unittest import TestCase


class SimpleState(object):

    StateVariable = namedtuple('StateVariable', 'name value')

    def __init__(self, state, regime, component_class):
        self.component_class = component_class
        self.state = state
        self.regime = regime

    def in_si_units(self):
        return self

    @property
    def variables(self):
        return (self.StateVariable(*i) for i in self.state.items())


# class TestDynamics(TestCase):

#     def test_liaf(self, dt=0.001 * un.ms, duration=100.0 * un.ms):

dt = 0.001 * un.ms
duration = 100.0 * un.ms

definition = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                'PyNNLeakyIntegrateAndFire')
properties = ninemlcatalog.load('neuron/LeakyIntegrateAndFire',
                                'PyNNLeakyIntegrateAndFireProperties')
initial_state = SimpleState(
    {'v': -65.0 * un.mV, 'end_refractory': 0.0 * un.s},
    'subthreshold', definition)
input_signal = AnalogSource.step(1 * un.ms, 50 * un.ms, 100 * un.ms,
                                 dt, 20 * un.ms)
dynamics = Dynamics(definition, properties, initial_state, start_t=0.0 * un.s,
                    dt=dt)
input_signal.connect_to(dynamics.port('i_synaptic'), delay=0.1 * un.ms)
dynamics.simulate(duration)
