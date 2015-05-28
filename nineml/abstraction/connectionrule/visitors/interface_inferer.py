from ...componentclass.visitors import ComponentClassInterfaceInferer
from .base import ConnectionRuleActionVisitor


class ConnectionRuleInterfaceInferer(ComponentClassInterfaceInferer,
                                        ConnectionRuleActionVisitor):

    """
    Not extended from base classes currently, just mixes in the connectionrule-
    specific action visitor to the component-class interface inferer.
    """
    pass
