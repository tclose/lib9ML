"""
docstring needed

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
import os
from urllib2 import urlopen
from lxml import etree
from itertools import chain
from nineml.xmlns import E
from . import ComponentVisitor
from ...expressions import Alias, Constant
from nineml.abstraction.componentclass.base import Parameter
from nineml.annotations import annotate_xml, read_annotations
from nineml.utils import expect_single, filter_expect_single
from nineml.xmlns import NINEML, MATHML, nineml_namespace
from nineml.exceptions import NineMLRuntimeError, handle_xml_exceptions


class ComponentClassXMLLoader(object):

    """This class is used by XMLReader internally.

    This class loads a NineML XML tree, and stores
    the components in ``components``. It o records which file each XML node
    was loaded in from, and stores this in ``component_srcs``.

    """

    def __init__(self, document=None):
        if document is None:
            document = Document()
        self.document = document

    def load_connectports(self, element):
        return element.attrib['source'], element.attrib['sink']

    @read_annotations
    @handle_xml_exceptions
    def load_parameter(self, element):
        return Parameter(name=element.attrib['name'],
                         dimension=self.document[element.attrib['dimension']])

    @read_annotations
    @handle_xml_exceptions
    def load_alias(self, element):
        name = element.attrib['name']
        rhs = self.load_single_internmaths_block(element)
        return Alias(lhs=name, rhs=rhs)

    @read_annotations
    @handle_xml_exceptions
    def load_constant(self, element):
        return Constant(name=element.attrib['name'],
                        value=float(element.text),
                        units=self.document[element.attrib['units']])

    def load_single_internmaths_block(self, element, checkOnlyBlock=True):
        if checkOnlyBlock:
            elements = list(element.iterchildren(tag=etree.Element))
            if len(elements) != 1:
                raise NineMLRuntimeError(
                    "Unexpected tags found '{}'"
                    .format("', '".join(e.tag for e in elements)))
        assert (len(element.findall(MATHML + "MathML")) +
                len(element.findall(NINEML + "MathInline"))) == 1
        if element.find(NINEML + "MathInline") is not None:
            mblock = expect_single(
                element.findall(NINEML + 'MathInline')).text.strip()
        elif element.find(MATHML + "MathML") is not None:
            mblock = self.load_mathml(
                expect_single(element.find(MATHML + "MathML")))
        return mblock

    def load_mathml(self, mathml):
        raise NotImplementedError

    def _load_blocks(self, element, block_names):
        """
        Creates a dictionary that maps class-types to instantiated objects
        """
        # Initialise loaded objects with empty lists
        loaded_objects = dict((block, []) for block in block_names)

        for t in element.iterchildren(tag=etree.Element):
            # Strip namespace
            tag = t.tag[len(NINEML):] if t.tag.startswith(NINEML) else t.tag
            if tag not in block_names:
                raise NineMLRuntimeError(
                    "Unexpected element {} was found in {} block in '{}'"
                    " (expected tags are: '{}')"
                    .format(tag, element.tag[len(NINEML):], self.document.url,
                            "', '".join(block_names)))
            loaded_objects[tag].append(self.tag_to_loader[tag](self, t))
        return loaded_objects

    tag_to_loader = {
        "Parameter": load_parameter,
        "Alias": load_alias,
        "Constant": load_constant
    }


class ComponentClassXMLWriter(ComponentVisitor):

    @annotate_xml
    def visit_parameter(self, parameter):
        return E(Parameter.element_name,
                 name=parameter.name,
                 dimension=parameter.dimension.name)

    @annotate_xml
    def visit_alias(self, alias):
        return E(Alias.element_name,
                 E("MathInline", alias.rhs_xml),
                 name=alias.lhs)

    @annotate_xml
    def visit_constant(self, constant):
        return E('Constant', str(constant.value),
                 name=constant.name,
                 units=constant.units.name)

from nineml.document import Document
