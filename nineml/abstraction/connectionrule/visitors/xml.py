"""
docstring needed

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from nineml.annotations import annotate_xml
from nineml.xmlns import E
from nineml.annotations import read_annotations
from ...componentclass.visitors.xml import (
    ComponentClassXMLLoader, ComponentClassXMLWriter)
from nineml.exceptions import handle_xml_exceptions
from ...expressions import Alias, Constant


#from nineml.abstraction_layer import ConnectionRule

class ConnectionRuleXMLLoader(ComponentClassXMLLoader):

    """This class is used by XMLReader internally.
    This class loads a NineML XML tree, and stores
    the components in ``components``. It records which file each XML node
    was loaded in from, and stores this in ``component_srcs``.
    """

    @read_annotations
    @handle_xml_exceptions
    def load_connectionruleclass(self, element, **kwargs):  # @UnusedVariable
        block_names = ('Parameter',)
        blocks = self._load_blocks(element, block_names=block_names)
        return ConnectionRule(
            name=element.attrib['name'],            
            propertyrecieveport=blocks["PropertyRecievePort"],
            parameters=blocks["Parameter"],
            constant=blocks["Constant"],
            alias=blocks["Alias"],
            select=blocks["Select"]
            )
    
    @read_annotations
    def load_select(self, element):
        block_names = ('Select')
        blocks = self.load_blocks(element,block_names=block_names)

        return Select(  
            mask=blocks["mask"],
            number=blocks["number"],#Does the appropriate object get expanded here
            preference=blocks["preference"],
            was_selecteds=blocks["was_selected"], 
            number_selected=blocks["number_selected"],
            random_variables=blocks["random_variables"], 
            select=blocks["select"], 
            repeat_whiles=blocks["repeat_while"])
            )

    @read_annotations
    def load_alias(self, element):
        name = element.attrib["name"]
        rhs = self.load_single_internmaths_block(element)
        return Alias(lhs=name, rhs=rhs)

    @read_annotations
    def load_constant(self, element):
        return Constant(
            name=element.attrib['name'],
            value=float(element.text),
            units=self.document[element.get('units')]
            )


    tag_to_loader = dict(
        tuple(ComponentClassXMLLoader.tag_to_loader.iteritems()) +
        (("ConnectionRule", load_connectionruleclass),
         ("Select", load_select),
         ("Alias", load_alias),
         ("Constant", load_constant)
        ))


class ConnectionRuleXMLWriter(ComponentClassXMLWriter):

    @annotate_xml
    def visit_componentclass(self, component_class, **kwargs):  # @UnusedVariable @IgnorePep8
        return E('ConnectionRule',
                 *self._sort(e.accept_visitor(self) for e in component_class),
                 name=component_class.name)
    
    @annotate_xml
    def visit_select(self, parameter):
        return E(Select.element_name,
                 name=parameter.name,
                 dimension=parameter.dimension.name)

    @annotate_xml
    def visit_alias(self, alias):
        return E(Alias.element_name,
                 E("MathInline", alias.rhs_cstr),
                 name=alias.lhs)

    @annotate_xml
    def visit_constant(self, constant):
        return E('Constant', str(constant.value),
                 name=constant.name,
                 units=constant.units.name)

from ..base import ConnectionRule
