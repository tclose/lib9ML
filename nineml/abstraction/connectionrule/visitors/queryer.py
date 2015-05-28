"""
Definitions for the ComponentQuery Class

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""

from itertools import chain
from ...componentclass.visitors.queryer import ComponentQueryer


class ConnectionRuleQueryer(ComponentQueryer):

    """
    ConnectionRuleQueryer provides a way of adding methods to query a
    ComponentClass object, without polluting the class
    """
    pass
