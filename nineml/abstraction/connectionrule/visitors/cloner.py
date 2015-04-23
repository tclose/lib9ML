"""
docstring needed

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from ...componentclass.visitors.cloner import (
    ComponentExpandPortDefinition, ComponentExpandAliasDefinition,
    ComponentCloner)
from .base import ConnectionRuleActionVisitor


class ConnectionRuleExpandPortDefinition(ConnectionRuleActionVisitor,
                                         ComponentExpandPortDefinition):

    def action_number(self, number, **kwargs):  # @UnusedVariable
        number.rhs_name_transform_inplace(self.namemap)

    def action_mask(self, mask, **kwargs):  # @UnusedVariable
        mask.rhs_name_transform_inplace(self.namemap)

    def action_preference(self, preference, **kwargs):  # @UnusedVariable
        preference.rhs_name_transform_inplace(self.namemap)

    def action_repeatwhile(self, repeatwhile, **kwargs):  # @UnusedVariable
        repeatwhile.rhs_name_transform_inplace(self.namemap)


class ConnectionRuleExpandAliasDefinition(ConnectionRuleActionVisitor,
                                          ComponentExpandAliasDefinition):

    """
    An action-class that walks over a component_class, and expands an alias in
    Aliases
    """

    def action_number(self, number, **kwargs):  # @UnusedVariable
        number.rhs_name_transform_inplace(self.namemap)

    def action_mask(self, mask, **kwargs):  # @UnusedVariable
        mask.rhs_name_transform_inplace(self.namemap)

    def action_preference(self, preference, **kwargs):  # @UnusedVariable
        preference.rhs_name_transform_inplace(self.namemap)

    def action_repeatwhile(self, repeatwhile, **kwargs):  # @UnusedVariable
        repeatwhile.rhs_name_transform_inplace(self.namemap)


class ConnectionRuleCloner(ComponentCloner):

    def visit_componentclass(self, component_class, **kwargs):
        super(ConnectionRuleCloner, self).visit_componentclass(component_class)
        ccn = component_class.__class__(
            name=component_class.name,
            parameters=[p.accept_visitor(self, **kwargs)
                        for p in component_class.parameters],
            analog_ports=[p.accept_visitor(self, **kwargs)
                          for p in component_class.analog_ports],
            event_ports=[p.accept_visitor(self, **kwargs)
                         for p in component_class.event_ports],
            select=component_class.select.accept_visitor(self),
            aliases=[
                a.accept_visitor(self, **kwargs)
                for a in component_class.aliases],
            constants=[c.accept_visitor(self, **kwargs)
                       for c in component_class.constants])
        return ccn
