import numpy.random
from nineml.user.component import Component


class RandomDistributionProperties(Component):
    """
    Component representing a random number randomdistribution, e.g. normal,
    gamma, binomial.

    *Example*::

        example goes here
    """
    nineml_type = 'RandomDistributionProperties'

    UNCERTML_PREFIX = 'http://www.uncertml.org/distributions/'

    default_funcs = {
        'uniform': numpy.random.uniform,
        'binomial': numpy.random.uniform,
        'poisson': numpy.random.poisson,
        'exponential': numpy.random.exponential,
        'normal': numpy.random.normal}

    @property
    def standard_library(self):
        return self.component_class.standard_library

    def get_nineml_type(self):
        return self.nineml_type

    def sample(self):
        if self.standard_library == self.UNCERTML_PREFIX + 'uniform':
            value = numpy.random.uniform(self['minimum'].value,
                                         self['maximum'].value)
        elif self.standard_library == self.UNCERTML_PREFIX + 'binomial':
            value = numpy.random.binomial(self['numberOfTrials'].value,
                                          self['probabilityOfSuccess'].value)
        elif self.standard_library == self.UNCERTML_PREFIX + 'poisson':
            value = numpy.random.poisson(1.0 / self['rate'].value)
        elif self.standard_library == self.UNCERTML_PREFIX + 'exponential':
            value = numpy.random.exponential(self['rate'].value)
        elif self.standard_library == self.UNCERTML_PREFIX + 'normal':
            value = numpy.random.normal(self['mean'].value,
                                         self['variannce'].value)
        else:
            raise NotImplementedError(
                "'{}' distributions are not support yet"
                .format(self.standard_library))
        return value
