"""
docstring goes here

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from lxml import etree  # @UnusedImport
from lxml.builder import ElementMaker
from nineml.exceptions import (
    NineMLXMLAttributeError, NineMLXMLBlockError, NineMLRuntimeError)
import re
import nineml

NINEML_VERSION = 1.0

nineml_v1_ns = 'http://nineml.net/9ML/1.0'
nineml_v2_ns = 'http://nineml.net/9ML/2.0'

if NINEML_VERSION == 1.0:
    nineml_ns = nineml_v1_ns
elif NINEML_VERSION == 2.0:
    nineml_ns = nineml_v2_ns
else:
    assert False

NINEML = '{' + nineml_ns + '}'
NINEMLv1 = '{' + nineml_v1_ns + '}'
NINEMLv2 = '{' + nineml_v2_ns + '}'
ALL_NINEML = (NINEMLv1, NINEMLv2)
MATHML = "{http://www.w3.org/1998/Math/MathML}"
UNCERTML = "{http://www.uncertml.org/2.0}"

# Extracts the ns from an lxml element tag
ns_re = re.compile(r'(\{.*\})(.*)')

Ev1 = ElementMaker(namespace=nineml_v1_ns, nsmap={None: nineml_v1_ns})
Ev2 = ElementMaker(namespace=nineml_v2_ns, nsmap={None: nineml_v2_ns})
E = ElementMaker(namespace=nineml_ns, nsmap={None: nineml_ns})


def get_element_maker(version):
    if isinstance(version, int):
        version = float(version)
    version = str(version)
    if str(version) == '1.0':
        element_maker = Ev1
    elif str(version) == '2.0':
        element_maker = Ev2
    else:
        raise NineMLRuntimeError(
            "Unrecognised 9ML version {} (1.0".format(version))
    return element_maker


def extract_ns(tag_name):
    return ns_re.match(tag_name).group(1)


def strip_ns(tag_name):
    return ns_re.match(tag_name).group(2)


def from_child_elem(element, child_classes, document, multiple=False,
                    allow_reference=False, allow_none=False, within=None,
                    unprocessed_elems=None, multiple_within=False,
                    allowed_attrib=[], **kwargs):
    """
    Loads a child element from the element, matching the tag name to the
    appropriate class and calling its 'unserialize' method
    """
    # Ensure child_classes is an iterable
    if isinstance(child_classes, type):
        child_classes = (child_classes,)
    assert child_classes, "No child classes supplied"
    # Get the namespace of the element (i.e. NineML version)
    ns = extract_ns(element.tag)
    # Get the parent element of the child elements to parse. For example the
    # in Projection elements where pre and post synaptic population references
    # are enclosed within 'Pre' or 'Post' tags respectively
    if within:
        within_elems = element.findall(ns + within)
        if len(within_elems) == 1:
            parent = within_elems[0]
            if any(a not in allowed_attrib for a in parent.attrib):
                raise NineMLXMLAttributeError(
                    "{} in '{}' has '{}' attributes when {} are expected"
                    .format(identify_element(parent), document.url,
                            "', '".join(parent.attrib.iterkeys()),
                            allowed_attrib))
            if not multiple_within and len([
                    c for c in parent.getchildren()
                    if c.tag != ns + 'Annotations']) > 1:
                raise NineMLXMLBlockError(
                    "{} in '{}' is only expected to contain a single child "
                    "block, found {}"
                    .format(identify_element(parent), document.url,
                            ", ".join(e.tag for e in parent.getchildren())))
            if unprocessed_elems:
                unprocessed_elems[0].discard(parent)
        elif not within_elems:
            if allow_none:
                return None
            else:
                raise NineMLXMLBlockError(
                    "Did not find {} block within {} element in '{}'"
                    .format(within, identify_element(element), document.url))
        else:
            raise NineMLXMLBlockError(
                "Found unexpected multiple {} blocks within {} in '{}'"
                .format(within, identify_element(element), document.url))
    else:
        parent = element
    # Get the list of child class names for error messages
    child_cls_names = "', '".join(c.nineml_type for c in child_classes)
    # Append all child classes
    children = []
    if allow_reference != 'only':
        for child_cls in child_classes:
            if ns == NINEMLv1:
                try:
                    tag_name = child_cls.v1_nineml_type
                except AttributeError:
                    tag_name = child_cls.nineml_type
            else:
                tag_name = child_cls.nineml_type
            for child_elem in parent.findall(ns + tag_name):
                children.append(child_cls.unserialize(child_elem, document,
                                                      **kwargs))
                if unprocessed_elems and not within:
                    unprocessed_elems[0].discard(child_elem)
    if allow_reference:
        for ref_elem in parent.findall(
                ns + nineml.reference.Reference.nineml_type):
            ref = nineml.reference.Reference.unserialize(ref_elem, document,
                                                         **kwargs)
            if isinstance(ref.user_object, child_classes):
                children.append(ref.user_object)
                if unprocessed_elems and not within:
                    unprocessed_elems[0].discard(ref_elem)
    if not children:
        if allow_none:
            result = [] if multiple else None
        else:
            raise NineMLXMLBlockError(
                "Did not find any child blocks with the tag{s} "
                "'{child_cls_names}'in the {parent_name} in '{url}'"
                .format(s=('s' if len(child_classes) else ''),
                        child_cls_names=child_cls_names,
                        parent_name=identify_element(parent),
                        url=document.url))
    elif multiple:
        result = children
    elif len(children) == 1:
        result = children[0]  # Expect single
    else:
        raise NineMLXMLBlockError(
            "Multiple children of types '{}' found within {} in '{}'"
            .format(child_cls_names, identify_element(parent), document.url))
    return result


def get_elem_attr(element, name, document, unprocessed_elems=None,
                  in_block=False, within=None, dtype=str, **kwargs):  # @UnusedVariable @IgnorePep8
    """
    Gets an attribute from an xml element with exception handling
    """
    if in_block:
        sub_elem = get_subblock(element, name, unprocessed_elems, document)
        attr_str = sub_elem.text
    else:
        if within is not None:
            elem = get_subblock(element, within, unprocessed_elems, document)
        else:
            elem = element
        try:
            attr_str = elem.attrib[name]
            if unprocessed_elems:
                unprocessed_elems[1].discard(name)
        except KeyError, e:
            try:
                return kwargs['default']
            except KeyError:
                raise NineMLXMLAttributeError(
                    "{} in '{}' is missing the {} attribute (found '{}' "
                    "attributes)".format(
                        identify_element(elem), document.url, e,
                        "', '".join(elem.attrib.iterkeys())))
    try:
        attr = dtype(attr_str)
    except ValueError, e:
        if isinstance(e, NineMLRuntimeError):
            raise
        else:
            raise NineMLXMLAttributeError(
                "'{}' attribute of {} in '{}', {}, cannot be converted to {} "
                "type".format(name, identify_element(element), document.url,
                              attr_str, dtype))
    return attr


def get_subblock(element, name, unprocessed_elems, document, **kwargs):  # @UnusedVariable @IgnorePep8
    ns = extract_ns(element.tag)
    found = element.findall(ns + name)
    if len(found) == 1:
        if unprocessed_elems:
            unprocessed_elems[0].discard(found[0])
    elif not found:
        raise NineMLXMLBlockError(
            "Did not find and child blocks with the tag '{}' within {} in "
            "'{url}'".format(name, identify_element(element),
                             url=document.url))
    else:
        raise NineMLXMLBlockError(
            "Found multiple child blocks with the tag '{}' within {} in "
            "'{url}'".format(name, identify_element(element),
                             url=document.url))
    return found[0]


def get_subblocks(element, name, unprocessed_elems, **kwargs):  # @UnusedVariable @IgnorePep8
    ns = extract_ns(element.tag)
    children = element.findall(ns + name)
    for child in children:
        if unprocessed_elems:
            unprocessed_elems[0].discard(child)
    return children


def identify_element(element):
    """
    Identifies an XML element for use in error messages
    """
    # Get the namespace of the element (i.e. NineML version)
    ns = extract_ns(element.tag)
    # Get the name of the element for error messages if present
    identity = element.tag[len(ns):]
    try:
        name = element.attrib['name']
    except KeyError:
        try:
            name = element.attrib['symbol']
        except KeyError:
            return identity
    return "'{}' {}".format(name, identity)


def unprocessed(unserialize):
    def unserialize_with_exception_handling(cls, element, *args, **kwargs):  # @UnusedVariable @IgnorePep8
        # Get the document object for error messages
        if args or 'document' in kwargs:  # if UL classmethod
            if args:
                document = args[0]
            else:
                document = kwargs['document']
            ns = extract_ns(element.tag)
            if ns == NINEMLv1:
                try:
                    nineml_type = cls.v1_nineml_type
                except AttributeError:
                    nineml_type = cls.nineml_type
            else:
                nineml_type = cls.nineml_type
            # Check the tag of the element matches the class names
            try:
                assert element.tag == (ns + nineml_type), (
                    "Found '{}' element, expected '{}'"
                    .format(element.tag, ns + nineml_type))
            except:
                raise
        else:
            document = cls.document  # if AL visitor method
        # Keep track of which blocks and attributes were processed within the
        # element
        unprocessed_elems = (set(e for e in element.getchildren()
                           if not isinstance(e, etree._Comment)),
                       set(element.attrib.iterkeys()))
        # The decorated method
        obj = unserialize(cls, element, *args, unprocessed_elems=unprocessed_elems,
                          **kwargs)
        # Check to see if there were blocks that were unprocessed_elems in the
        # element
        blocks, attrs = unprocessed_elems
        if blocks:
            raise NineMLXMLBlockError(
                "Found unrecognised block{s} '{remaining}' within "
                "{elem_name} in '{url}'"
                .format(s=('s' if len(blocks) > 1 else ''),
                        remaining="', '".join(str(b.tag) for b in blocks),
                        elem_name=identify_element(element), url=document.url))
        if attrs:
            raise NineMLXMLAttributeError(
                "Found unrecognised attribute{s} '{remaining}' within "
                "{elem_name} in '{url}'"
                .format(s=('s' if len(attrs) > 1 else ''),
                        remaining="', '".join(attrs),
                        elem_name=identify_element(element), url=document.url))
        return obj
    return unserialize_with_exception_handling
