from ...componentclass.visitors import ComponentClassInterfaceInferer
from .base import RandomDistributionActionVisitor


class RandomDistributionInterfaceInferer(ComponentClassInterfaceInferer,
                                        RandomDistributionActionVisitor):

    """
    Not extended from base classes currently, just mixes in the randomdistribution-
    specific action visitor to the component-class interface inferer.
    """
    pass
