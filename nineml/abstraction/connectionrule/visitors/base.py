"""
docstring needed

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""


from ...componentclass.visitors import (
    ComponentActionVisitor, ComponentElementFinder)



class ConnectionRuleActionVisitor(ComponentActionVisitor):

    def visit_componentclass(self, component_class, **kwargs):
        super(ConnectionRuleActionVisitor, self).visit_componentclass(
            component_class, **kwargs)

    def visit_number(self, number, **kwargs):
        self.action_number(number, **kwargs)

    def visit_mask(self, mask, **kwargs):
        self.action_mask(mask, **kwargs)

    def visit_preference(self, preference, **kwargs):
        self.action_preference(preference, **kwargs)

    def visit_repeatwhile(self, repeatwhile, **kwargs):
        self.action_repeatwhile(repeatwhile, **kwargs)

    def visit_selected(self, selected, **kwargs):
        self.action_selected(selected, **kwargs)

    def visit_numberselected(self, numberselected, **kwargs):
        self.action_numberselected(numberselected, **kwargs)

    def visit_select(self, select, **kwargs):
        self.action_select(select, **kwargs)
        self._scopes.append(select)
        for e in select:
            e.accept_visitor(self, **kwargs)
        popped = self._scopes.pop()
        assert popped is select
