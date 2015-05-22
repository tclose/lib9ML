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
        block_names = ('Parameter', 'PropertyRecievePort', 'Constant',
                       'Alias', 'Select')
        blocks = self._load_blocks(element, block_names=block_names)
        return ConnectionRule(
            name=element.attrib['name'],            
            propertyrecieveport=blocks["PropertyRecievePort"],
            parameters=blocks["Parameter"],
            constant=blocks["Constant"],
            alias=blocks["Alias"],
            select=blocks["Select"])
    
    @read_annotations
    def load_select(self, element):
        block_names = ('Mask', 'Number', 'Preference', 'Selected',
                       'NumberSelected', 'RandomVariables', 'Select', 'RepeatUntil')
        blocks = self.load_blocks(element, block_names=block_names)
        return Select(  
            mask=blocks["Mask"],
            number=blocks["Number"],#Does the appropriate object get expanded here
            preference=blocks["Preference"],
            selecteds=blocks["Selected"], 
            number_selecteds=blocks["NumberSelected"],
            random_variables=blocks["RandomVariables"], 
            select=blocks["Select"], 
            repeat_untils=blocks["RepeatUntil"])


    tag_to_loader = dict(
        tuple(ComponentClassXMLLoader.tag_to_loader.iteritems()) +
        (("ConnectionRule", load_connectionruleclass),
         ("Select", load_select)))


class ConnectionRuleXMLWriter(ComponentClassXMLWriter):

    @annotate_xml
    def visit_componentclass(self, component_class, **kwargs):  # @UnusedVariable @IgnorePep8
        return E('ConnectionRule',
                 *self._sort(e.accept_visitor(self) for e in component_class),
                 name=component_class.name)

    @annotate_xml
    def visit_select(self, select):
        return E.Select(name=select.name)


from ..base import ConnectionRule
from ..select import Select
