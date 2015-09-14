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
from nineml.abstraction.ports import PropertyReceivePort
from nineml.annotations import annotate_xml, read_annotations


class ConnectionRule(ComponentClass):

    element_name = 'ConnectionRule'
    defining_attributes = ('name', '_parameters', '_select', '_constants',
                           '_aliases')
    class_to_member_dict = {PropertyReceivePort: '_property_receive_ports'}

    def __init__(self, name, select, parameters=None, constants=None,
                 aliases=None):
        super(ConnectionRule, self).__init__(
            name=name, parameters=parameters, aliases=aliases,
            constants=constants)
        self._select = select

    @property
    def select(self):
        return self._select

    @property
    def property_receive_ports(self):
        return self._property_receive_ports.itervalues()

    @property
    def property_receive_port_names(self):
        return self._property_receive_ports.iterkeys()

    def property_receive_port(self, name):
        return self._property_receive_ports[name]

    @property
    def selects(self):
        """
        Iterate through nested select statements
        """
        select = self._select
        yield select
        if select.select is not None:
            select = self._select
            yield select
        else:
            raise StopIteration

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

    def dimension_of(self, element):
        try:
            resolver = self._dimension_resolver
        except AttributeError:  # If dimension resolver hasn't been set
            resolver = ConnectionRuleDimensionResolver(self)
            self._dimension_resolver = resolver
        return resolver.dimension_of(element)

    def _find_element(self, element):
        return ConnectionRuleElementFinder(element).found_in(self)

    def validate(self):
        ConnectionRuleValidator.validate_componentclass(self)

    @property
    def all_expressions(self):
        extractor = ConnectionRuleExpressionExtractor()
        extractor.visit(self)
        return extractor.expressions

    def to_xml(self, document, **kwargs):  # @UnusedVariable
        self.standardize_unit_dimensions()
        self.validate()
        return ConnectionRuleXMLWriter(document).visit(self)

    @classmethod
    def from_xml(cls, element, document, **kwargs):  # @UnusedVariable
        return ConnectionRuleXMLLoader(document).load_connectionruleclass(
            element)


from .visitors.cloner import ConnectionRuleCloner
from .visitors.modifiers import (
    ConnectionRuleRenameSymbol, ConnectionRuleAssignIndices)
from .visitors.queriers import (
    ConnectionRuleRequiredDefinitions, ConnectionRuleElementFinder,
    ConnectionRuleExpressionExtractor, ConnectionRuleDimensionResolver)
from .visitors.validators import ConnectionRuleValidator
from .visitors.xml import (
    ConnectionRuleXMLLoader, ConnectionRuleXMLWriter)
