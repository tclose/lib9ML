#!/usr/bin/env python
"""
docstring goes here

.. module:: connection_generator.py
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: Mikael Djurfeldt <mikael.djurfeldt@incf.org>
.. moduleauthor:: Dragan Nikolic <dnikolic@incf.org>

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from ..componentclass import ComponentClass


class ConnectionRuleClass(ComponentClass):

    element_name = 'ConnectionRuleClass'
    defining_attributes = ('name', '_parameters', 'standard_library')

    def __init__(self, name, standard_library, parameters=None):
        super(ConnectionRuleClass, self).__init__(
            name, parameters)
        self.standard_library = standard_library

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_componentclass(self, **kwargs)

    def __copy__(self, memo=None):  # @UnusedVariable
        return ConnectionRuleCloner().visit(self)

    def rename_symbol(self, old_symbol, new_symbol):
        ConnectionRuleRenameSymbol(self, old_symbol, new_symbol)

    def assign_indices(self):
        ConnectionRuleAssignIndices(self)

    def required_for(self, expressions):
        return ConnectionRuleRequiredDefinitions(self, expressions)

    def _find_element(self, element):
        return ConnectionRuleElementFinder(element).found_in(self)

    def validate(self):
        ConnectionRuleValidator.validate_componentclass(self)

from .utils.cloner import ConnectionRuleCloner
from .utils.modifiers import (
    ConnectionRuleRenameSymbol, ConnectionRuleAssignIndices)
from .utils.visitors import (
    ConnectionRuleRequiredDefinitions, ConnectionRuleElementFinder)
from .validators import ConnectionRuleValidator
