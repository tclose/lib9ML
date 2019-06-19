from builtins import zip
from builtins import range
from copy import deepcopy
from itertools import chain, product
import math
from itertools import repeat
from nineml.exceptions import NineMLUsageError
from nineml.user.component import Component


class ConnectionRuleProperties(Component):
    """
    docstring needed
    """
    nineml_type = 'ConnectionRuleProperties'

    def get_nineml_type(self):
        return self.nineml_type

    @property
    def standard_library(self):
        return self.component_class.standard_library

    @property
    def lib_type(self):
        return self.component_class.lib_type


class Connections(object):
    """
    A group of connections between a source and destination array, sampled
    from a connection rule properties
    """

    def __init__(self, rule_properties, source_size,
                 destination_size, random_state, **kwargs):  # @UnusedVariable
        if (rule_properties.lib_type == 'OneToOne' and
                source_size != destination_size):
            raise NineMLUsageError(
                "Cannot connect to populations of different sizes "
                "({} and {}) with OneToOne connection rule"
                .format(source_size, destination_size))
        if not isinstance(rule_properties, ConnectionRuleProperties):
            raise NineMLUsageError(
                "'rule_properties' argument ({}) must be a "
                "ConnectcionRuleProperties instance".format(rule_properties))
        self._rule_properties = rule_properties
        self._source_size = source_size
        self._destination_size = destination_size
        self._state = random_state

    def __eq__(self, other):
        try:
            return (self._rule_properties == other._rule_properties and
                    self._source_size == other._source_size and
                    self._destination_size == other._destination_size)
        except AttributeError:
            return False

    @property
    def rule_properties(self):
        return self._rule_properties

    @property
    def rule(self):
        return self.rule_properties.component_class

    @property
    def lib_type(self):
        return self.rule_properties.lib_type

    @property
    def source_size(self):
        return self._source_size

    @property
    def destination_size(self):
        return self._destination_size

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return ("{}(rule={}, src_size={}, dest_size={})"
                .format(self.__class__.__name__, self.lib_type,
                        self.source_size, self.destination_size))

    def __iter__(self):
        """
        Returns an iterator over all the source/destination index pairings
        with a connection.
        `src`  -- the indices to get the connections from
        `dest` -- the indices to get the connections to
        """
        # Create a snapshot of the random state in order to exactly recreate
        # the conn generator if required
        state = self._state
        self._state = deepcopy(self._state)
        if self.lib_type == 'AllToAll':
            conn = self._all_to_all()
        elif self.lib_type == 'OneToOne':
            conn = self._one_to_one()
        elif self.lib_type == 'Explicit':
            conn = self._explicit_connection_list()
        elif self.lib_type == 'Probabilistic':
            conn = self._probabilistic_connectivity(state)
        elif self.lib_type == 'RandomFanIn':
            conn = self._random_fan_in(state)
        elif self.lib_type == 'RandomFanOut':
            conn = self._random_fan_out(state)
        else:
            assert False
        return conn

    def inverse(self):
        return ((j, i) for i, j in self)

    def _all_to_all(self):  # @UnusedVariable
        return product(range(self._source_size),
                       range(self._destination_size))

    def _one_to_one(self):  # @UnusedVariable
        assert self._source_size == self._destination_size
        return ((i, i) for i in range(self._source_size))

    def _explicit_connection_list(self):  # @UnusedVariable
        return zip(
            self._rule_properties.property('sourceIndices').value.values,
            self._rule_properties.property('destinationIndices').value.values)

    def _probabilistic_connectivity(self, state):  # @UnusedVariable
        # Reinitialize the connectivity generator with the same RNG so that
        # it selects the same numbers
        p = float(self._rule_properties.property('probability').value)
        # Get an iterator over all of the source dest pairs to test
        return chain(*(
            ((s, d) for d in range(self._destination_size) if state.rand() < p)
            for s in range(self._source_size)))

    def _random_fan_in(self, state):  # @UnusedVariable
        N = int(self._rule_properties.property('number').value)
        return chain(*(
            zip((int(math.floor(state.rand() * self._source_size))
                 for _ in range(N)), repeat(d))
            for d in range(self._destination_size)))

    def _random_fan_out(self, state):  # @UnusedVariable
        N = int(self._rule_properties.property('number').value)
        return chain(*(
            zip(repeat(s),
                (int(math.floor(state.rand() * self._destination_size))
                 for _ in range(N)))
            for s in range(self._source_size)))

    @property
    def key(self):
        return '{}__{}__{}__{}'.format(self.rule_properties.name,
                                       self.source_size,
                                       self.destination_size,
                                       self._seed)
