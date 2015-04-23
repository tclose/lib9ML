from .base import ConnectionRuleActionVisitor
from ...componentclass.visitors.queriers import (
    ComponentRequiredDefinitions, ComponentClassInterfaceInferer,
    ComponentElementFinder, ComponentExpressionExtractor,
    ComponentDimensionResolver)


class ConnectionRuleInterfaceInferer(ComponentClassInterfaceInferer,
                                        ConnectionRuleActionVisitor):

    """
    Not extended from base classes currently, just mixes in the connectionrule-
    specific action visitor to the component-class interface inferer.
    """
    pass


class ConnectionRuleRequiredDefinitions(ComponentRequiredDefinitions,
                                        ConnectionRuleActionVisitor):

    def __init__(self, component_class, expressions):
        ConnectionRuleActionVisitor.__init__(self,
                                             require_explicit_overrides=False)
        ComponentRequiredDefinitions.__init__(self, component_class,
                                              expressions)


class ConnectionRuleElementFinder(ComponentElementFinder,
                                  ConnectionRuleActionVisitor):

    def __init__(self, element):
        ConnectionRuleActionVisitor.__init__(self,
                                             require_explicit_overrides=True)
        ComponentElementFinder.__init__(self, element)

    def visit_number(self, number, **kwargs):  # @UnusedVariable
        if self.element is number:
            self._found()

    def visit_mask(self, mask, **kwargs):  # @UnusedVariable
        if self.element is mask:
            self._found()

    def visit_preference(self, preference, **kwargs):  # @UnusedVariable
        if self.element is preference:
            self._found()

    def visit_repeatwhile(self, repeatwhile, **kwargs):  # @UnusedVariable
        if self.element is repeatwhile:
            self._found()

    def visit_selected(self, selected, **kwargs):  # @UnusedVariable
        if self.element is selected:
            self._found()

    def visit_numberselected(self, numberselected, **kwargs):  # @UnusedVariable @IgnorePep8
        if self.element is numberselected:
            self._found()

    def visit_select(self, select, **kwargs):  # @UnusedVariable
        if self.element is select:
            self._found()


class ConnectionRuleExpressionExtractor(ComponentExpressionExtractor,
                                        ConnectionRuleActionVisitor):

    def __init__(self):
        ConnectionRuleActionVisitor.__init__(self,
                                             require_explicit_overrides=True)
        ComponentExpressionExtractor.__init__(self)


class ConnectionRuleDimensionResolver(ComponentDimensionResolver,
                                      ConnectionRuleActionVisitor):
    pass
