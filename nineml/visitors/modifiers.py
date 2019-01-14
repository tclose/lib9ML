from .base import BaseVisitor
from nineml.units import Unit, common_units


class ToSIUnitsConvertor(BaseVisitor):
    """
    Converts all quantities in the object to SI units (i.e. unit.power = 0)
    """

    def __init__(self):
        self.si_units = {u.dimension: u for u in common_units if u.power == 0}

    def action_quantity(self, quantity, **kwargs):  # @UnusedVariable
        dim = quantity.units.dimension
        try:
            si_unit = self.si_units[dim]
        except KeyError:
            si_unit = Unit(dim.name + 'SIUnit', dim, power=0)
            self.si_units[dim] = si_unit
        quantity.set_units(si_unit)

    def default_action(self, obj, nineml_cls, **kwargs):  # @UnusedVariable
        pass
