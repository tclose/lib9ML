"""
docstring needed

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from lxml import etree
from . import ComponentVisitor
from ...expressions import Alias, Constant
from nineml.abstraction.componentclass.base import Parameter
from nineml.annotations import annotate, read_annotations
from nineml.serialize import (
    E, strip_ns, extract_ns, get_elem_attr, identify_element,
    un_proc_essed, ALL_NINEML, NINEMLv1)
from nineml.exceptions import NineMLXMLBlockError
from nineml.abstraction.expressions import Expression


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

    @read_annotations
    @un_proc_essed
    def load_parameter(self, element, **kwargs):  # @UnusedVariable
        return Parameter(name=get_elem_attr(element, 'name', self.document,
                                            **kwargs),
                         dimension=self.document[
                             get_elem_attr(element, 'dimension', self.document,
                                           **kwargs)])

    @read_annotations
    @un_proc_essed
    def load_alias(self, element, **kwargs):  # @UnusedVariable
        name = get_elem_attr(element, 'name', self.document, **kwargs)
        rhs = self.load_expression(element, **kwargs)
        return Alias(lhs=name, rhs=rhs)

    @read_annotations
    @un_proc_essed
    def load_constant(self, element, **kwargs):  # @UnusedVariable
        ns = extract_ns(element.tag)
        if ns == NINEMLv1:
            value = float(element.text)
        else:
            value = get_elem_attr(element, 'value', self.document,
                                  dtype=float, **kwargs)
        return Constant(
            name=get_elem_attr(element, 'name', self.document, **kwargs),
            value=value,
            units=self.document[
                get_elem_attr(element, 'units', self.document, **kwargs)])

    def load_expression(self, element, **kwargs):
        return get_elem_attr(element, 'MathInline', self.document,
                             in_block=True, dtype=Expression, **kwargs)

    def _load_blocks(self, element, block_names, unprocessed_elems=None,
                     prev_block_names={}, ignore=[], **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Creates a dictionary that maps class-types to instantiated objects
        """
        # Get the XML namespace (i.e. NineML version)
        ns = extract_ns(element.tag)
        assert ns in ALL_NINEML
        # Initialise loaded objects with empty lists
        loaded_objects = dict((block, []) for block in block_names)
        for t in element.iterchildren(tag=etree.Element):
            # Used in un_proc_essed decorator
            if unprocessed_elems:
                unprocessed_elems[0].discard(t)
            # Strip namespace
            tag = (t.tag[len(ns):]
                   if t.tag.startswith(ns) else t.tag)
            if (ns, tag) not in ignore:
                if tag not in block_names:
                    raise NineMLXMLBlockError(
                        "Unexpected block {} within {} in '{}', expected: {}"
                        .format(tag, identify_element(element),
                                self.document.url, ', '.join(block_names)))
                loaded_objects[tag].append(self.tag_to_loader[tag](self, t))
        return loaded_objects

    tag_to_loader = {
        "Parameter": load_parameter,
        "Alias": load_alias,
        "Constant": load_constant
    }


class ComponentClassXMLWriter(ComponentVisitor):

    def __init__(self, document, E, **options):
        self.document = document
        self.E = E
        self.options = options

    @property
    def ns(self):
        return self.E._namespace

    @annotate
    def visit_parameter(self, parameter, **kwargs):
        return self.E(Parameter.nineml_type,
                      name=parameter.name,
                      dimension=parameter.dimension.name)

    @annotate
    def visit_alias(self, alias, **kwargs):
        return self.E(Alias.nineml_type,
                      self.E("MathInline", alias.rhs_xml),
                      name=alias.lhs)

    @annotate
    def visit_constant(self, constant, **kwargs):
        if self.ns == NINEMLv1:
            xml = self.E(Constant.nineml_type,
                         repr(constant.value),
                         name=constant.name,
                         units=constant.units.name)
        else:
            xml = self.E(Constant.nineml_type,
                         name=constant.name,
                         value=repr(constant.value),
                         units=constant.units.name)
        return xml


from nineml.document import Document  # @IgnorePep8
