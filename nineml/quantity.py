from .base import BaseNineMLObject
from nineml.annotations import annotate_xml, read_annotations
from nineml.units import unitless, Unit
from nineml.exceptions import NineMLRuntimeError, NineMLMissingElementError
from nineml.values import (
    SingleValue, ArrayValue, RandomDistributionValue, BaseValue)
from nineml.xmlns import E, NINEML
from nineml.utils import expect_single


class Quantity(BaseNineMLObject):

    """
    Representation of a numerical- or string-valued parameter.

    A numerical parameter is a (name, value, units) triplet, a string parameter
    is a (name, value) pair.

    Numerical values may either be numbers, or a component_class that generates
    numbers, e.g. a RandomDistribution instance.
    """
    element_name = 'Quantity'

    defining_attributes = ("value", "units")

    def __init__(self, value, units=None):
        super(Quantity, self).__init__()
        if not isinstance(value, (SingleValue, ArrayValue,
                                  RandomDistributionValue)):
            try:
                # Convert value from float
                value = SingleValue(float(value))
            except TypeError:
                # Convert value from iterable
                value = ArrayValue(value)
        if units is None:
            units = unitless
        if not isinstance(units, Unit):
            raise Exception("Units ({}) must of type <Unit>".format(units))
        if isinstance(value, (int, float)):
            value = SingleValue(value)
        self._value = value
        self.units = units

    def __hash__(self):
        if self.is_single():
            hsh = hash(self.value) ^ hash(self.units)
        else:
            hsh = hash(self.units)
        return hsh

    def __iter__(self):
        """For conveniently expanding quantities like a tuple"""
        return (self.value, self.units)

    @property
    def value(self):
        return self._value

    def __getitem__(self, index):
        if self.is_array():
            return self._value.values[index]
        elif self.is_single():
            return self._value.value
        else:
            raise NineMLRuntimeError(
                "Cannot get item from random distribution")

    def set_units(self, units):
        if units.dimension != self.units.dimension:
            raise NineMLRuntimeError(
                "Can't change dimension of quantity from '{}' to '{}'"
                .format(self.units.dimension, units.dimension))
        self.units = units

    def __repr__(self):
        units = self.units.name
        if u"µ" in units:
            units = units.replace(u"µ", "u")
        return ("{}(value={}, units={})"
                .format(self.element_name, self.value, units))

    @annotate_xml
    def to_xml(self, **kwargs):  # @UnusedVariable
        return E(self.element_name,
                 self._value.to_xml(),
                 units=self.units.name)

    @classmethod
    @read_annotations
    def from_xml(cls, element, document, **kwargs):  # @UnusedVariable
        value = BaseValue.from_parent_xml(
            expect_single(element.findall(NINEML, document, **kwargs)))
        try:
            units_str = element.attrib['units']
        except KeyError:
            raise NineMLRuntimeError(
                "{} element '{}' is missing 'units' attribute (found '{}')"
                .format(element.tag, element.get('name', ''),
                        "', '".join(element.attrib.iterkeys())))
        try:
            units = document[units_str]
        except KeyError:
            raise NineMLMissingElementError(
                "Did not find definition of '{}' units in the current "
                "document.".format(units_str))
        return cls(value=value, units=units)
