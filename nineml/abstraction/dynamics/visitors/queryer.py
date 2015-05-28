"""
Definitions for the ComponentQuery Class

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""

from itertools import chain
from nineml.utils import filter_expect_single
from ...componentclass.visitors.queryer import ComponentQueryer


class DynamicsQueryer(ComponentQueryer):

    """
    DynamicsQueryer provides a way of adding methods to query a
    ComponentClass object, without polluting the class
    """

