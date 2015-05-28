"""
Definitions for the ComponentQuery Class

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""

__all__ = ['ComponentQueryer']


class ComponentQueryer(object):

    """
    ComponentQueryer provides a way of adding methods to query a
    ComponentClass object, without polluting the class
    """

    def __init__(self, component_class):
        """Constructor for the ComponentQueryer"""
        self.component_class = component_class

