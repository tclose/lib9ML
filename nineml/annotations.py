from copy import copy
from collections import defaultdict
from itertools import chain
from nineml.xml import E, extract_xmlns, strip_xmlns
from nineml.base import DocumentLevelObject, BaseNineMLObject
import re
from nineml.xml import ElementMaker, nineml_ns, etree
from nineml.exceptions import (
    NineMLXMLError, NineMLRuntimeError, NineMLNameError)


def read_annotations(from_xml):
    def annotate_from_xml(cls, element, *args, **kwargs):
        nineml_xmlns = extract_xmlns(element.tag)
        annot_elem = expect_none_or_single(
            element.findall(nineml_xmlns + Annotations.nineml_type))
        if annot_elem is not None:
            # Extract the annotations
            annotations = Annotations.from_xml(annot_elem, **kwargs)
            # Get a copy of the element with the annotations stripped
            element = copy(element)
            element.remove(element.find(nineml_xmlns +
                                        Annotations.nineml_type))
        else:
            annotations = Annotations()
        if cls.__class__.__name__ == 'DynamicsXMLLoader':
            # FIXME: Hack until I work out the best way to let other 9ML
            #        objects ignore this kwarg TGC 6/15
            valid_dims = annotations.get(
                PY9ML_NS, VALIDATION, DIMENSIONALITY, default='True') == 'True'
            kwargs['validate_dimensions'] = valid_dims
        nineml_object = from_xml(cls, element, *args, **kwargs)
        nineml_object._annotations = annotations
        return nineml_object
    return annotate_from_xml


def annotate_xml(to_xml):
    def annotate_to_xml(self, document_or_obj, E=E, **kwargs):
        # If Abstraction Layer class
        if xml_visitor_module_re.match(type(self).__module__):
            obj = document_or_obj
            options = self.options
        # If User Layer class
        else:
            obj = self
            options = kwargs
        elem = to_xml(self, document_or_obj, E=E, **kwargs)
        if not options.get('no_annotations', False) and len(obj.annotations):
            elem.append(obj.annotations.to_xml(E=E, **kwargs))
        return elem
    return annotate_to_xml


class Annotations(DocumentLevelObject):
    """
    Is able to handle a basic hierarchical annotations format where the first
    level is the namespace of each sub element in the Annotations block
    """

    nineml_type = 'Annotations'
    defining_attributes = ('_namespaces',)

    def __init__(self, document=None):
        super(Annotations, self).__init__(document)
        self._namespaces = {}

    def __repr__(self):
        return "Annotations:\n{}".format(
            "\n".join(str(v) for v in self._namespaces.itervalues()))

    def __len__(self):
        return len(self._namespaces)

    def __getitem__(self, key):
        try:
            return self._namespaces[key]
        except KeyError:
            raise NineMLNameError(
                "'{}' namespace not in annotations".format(key))

    def __contains__(self, key):
        return key in self._namespaces

    def set(self, namespace, *args):
        try:
            ns = self[namespace]
        except KeyError:
            ns = self._namespaces[namespace] = _AnnotationsNamespace(namespace)
        ns.set(*args)

    def get(self, namespace, *args, **kwargs):
        try:
            return self[namespace].get(*args, **kwargs)
        except KeyError:
            if 'default' in kwargs:
                return kwargs['default']
            else:
                raise NineMLNameError(
                    "No annotation at path '{}'".format("', '".join(args)))

    def to_xml(self, E=E, **kwargs):  # @UnusedVariable
        members = []
        for ns, annot_ns in self._namespaces.iteritems():
            if isinstance(annot_ns, _AnnotationsNamespace):
                for branch in annot_ns.branches:
                    members.append(branch.to_xml(ns=ns, E=E, **kwargs))
            else:
                members.append(annot_ns)  # Append unprocessed XML
        return E(self.nineml_type, *members)

    @classmethod
    def from_xml(cls, element, annotations_ns=None, **kwargs):  # @UnusedVariable @IgnorePep8
        if annotations_ns is None:
            annotations_ns = []
        elif isinstance(annotations_ns, basestring):
            annotations_ns = [annotations_ns]
        assert strip_xmlns(element.tag) == cls.nineml_type
        annot = cls(**kwargs)
        for child in element.getchildren():
            ns = extract_xmlns(child.tag)
            if not ns:
                raise NineMLXMLError(
                    "All annotations must have a namespace: {}".format(
                        etree.tostring(child, pretty_print=True)))
            ns = ns[1:-1]  # strip braces
            if ns == nineml_ns or ns in annotations_ns:
                name = strip_xmlns(child.tag)
                try:
                    namespace = annot[ns]
                except KeyError:
                    annot._namespaces[ns] = _AnnotationsNamespace(ns)
                    namespace = annot._namespaces[ns]
                namespace[name] = _AnnotationsBranch.from_xml(child)
            else:
                annot._namespaces[ns] = child  # Don't process, just ignore
        return annot

    def _copy_to_clone(self, clone, memo, **kwargs):
        self._clone_defining_attr(clone, memo, **kwargs)
        clone._document = None

    def equals(self, other, **kwargs):  # @UnusedVariable
        try:
            if self.nineml_type != other.nineml_type:
                return False
        except AttributeError:
            return False
        if set(self._namespaces.keys()) != set(other._namespaces.keys()):
            return False
        for k, s in self._namespaces.iteritems():
            o = other._namespaces[k]
            if not isinstance(s, _AnnotationsNamespace):
                s = etree.tostring(s)
                o = etree.tostring(o)
            if s != o:
                return False
        return True


class _AnnotationsNamespace(BaseNineMLObject):
    """
    Like a defaultdict, but initialises AnnotationsBranch with a name
    """
    nineml_type = '_AnnotationsNamespace'
    defining_attributes = ('_ns', '_branches')

    def __init__(self, ns):
        self._ns = ns
        self._branches = {}

    @property
    def ns(self):
        return self._ns

    @property
    def branches(self):
        return self._branches.itervalues()

    def __repr__(self):
        rep = '"{}":\n'.format(self.ns)
        rep += '\n'.join(v._repr('  ') for v in self.branches)
        return rep

    def __getitem__(self, key):
        return self._branches[key]

    def __setitem__(self, key, val):
        if not isinstance(val, _AnnotationsBranch):
            raise NineMLRuntimeError(
                "Attempting to set directly to Annotations namespace '{}' "
                "(key={}, val={})".format(self._ns, key, val))
        self._branches[key] = val

    def set(self, key, *args):
        try:
            branch = self[key]
        except KeyError:
            branch = self._branches[key] = _AnnotationsBranch(key)
        branch.set(*args)

    def get(self, key, *args, **kwargs):
        try:
            return self[key].get(*args, **kwargs)
        except KeyError:
            if 'default' in kwargs:
                return kwargs['default']
            else:
                raise NineMLNameError(
                    "No annotation at path '{}'".format("', '".join(args)))

    def equals(self, other, **kwargs):  # @UnusedVariable
        try:
            if self.nineml_type != other.nineml_type:
                return False
        except AttributeError:
            return False
        return self._ns == other._ns and self._branches == other._branches


class _AnnotationsBranch(BaseNineMLObject):

    nineml_type = '_AnnotationsBranch'
    defining_attributes = ('_branches', '_attr', '_name')

    def __init__(self, name, attr=None, branches=None, text=None, ns=None):
        if attr is None:
            attr = {}
        if branches is None:
            branches = defaultdict(list)
        self._branches = branches
        self._name = name
        self._attr = attr
        self._text = text
        self._ns = ns

    @property
    def name(self):
        return self._name

    @property
    def text(self):
        return self._text

    @property
    def ns(self):
        return self._ns

    def __repr__(self):
        return self._repr()

    def equals(self, other, **kwargs):  # @UnusedVariable
        try:
            if self.nineml_type != other.nineml_type:
                return False
        except AttributeError:
            return False
        return (self._branches == other._branches and
                self._name == other._name and
                self._attr == other._attr)

    def _repr(self, indent=''):
        rep = "{}{}:".format(indent, self.name)
        if self._attr:
            rep += '\n' + '\n'.join('{}{}={}'.format(indent + '  ', *i)
                                    for i in self._attr.iteritems())
        if self._branches:
            rep += '\n' + '\n'.join(
                chain(*((b._repr(indent=indent + '  ') for b in key_branch)
                        for key_branch in self._branches.itervalues())))
        return rep

    def attr_values(self):
        return self._attr.itervalues()

    def attr_keys(self):
        return self._attr.iterkeys()

    def attr_items(self):
        return self._attr.iteritems()

    @property
    def branches(self):
        return self._branches.itervalues()

    def __iter__(self):
        return self._branches.keys()

    def __getitem__(self, key):
        if key in self._branches:
            key_branches = self._branches[key]
        else:
            raise NineMLNameError(
                "'{}' does not have branch or attribute '{}'"
                .format(self._name, key))
        return key_branches

    def set(self, key, *args):
        if not args:
            raise NineMLRuntimeError("No value was provided to set of '{}' "
                                     "in annotations branch '{}'"
                                     .format(key, self.name))
        if len(args) == 1:
            self._attr[key] = str(args[0])
        else:
            # Recurse into branches while there are remaining args
            key_branches = self._branches[key]
            if len(key_branches) == 1:
                branch = key_branches[0]
            elif not key_branches:
                branch = _AnnotationsBranch(key)
                key_branches.append(branch)
            else:
                raise NineMLNameError(
                    "Multiple branches found for key '{}' in annoations branch"
                    " '{}', cannot use 'set' method".format(
                        key, self._name))
            branch.set(*args)  # recurse into branch

    def get(self, key, *args, **kwargs):
        if not args:
            if 'default' in kwargs:
                val = self._attr.get(key, kwargs['default'])
            else:
                val = self._attr[key]
        else:
            if key in self._branches:
                key_branches = self._branches[key]
                if len(key_branches) == 1:
                    # Recurse into branches while there are remaining args
                    val = key_branches[0].get(*args, **kwargs)
                else:
                    raise NineMLNameError(
                        "Multiple branches found for key '{}' in annoations "
                        "branch '{}', cannot use 'get' method".format(
                            key, self._name))
            else:
                if 'default' in kwargs:
                    return kwargs['default']
                else:
                    raise NineMLNameError(
                        "No annotation at path '{}'".format("', '".join(args)))
        return val

    def to_xml(self, default_ns=None, E=E, **kwargs):  # @UnusedVariable
        nsmap = {}
        if default_ns is not None:
            nsmap[None] = default_ns
            ns = default_ns
        if self.ns is not None:
            ns = self.ns
        if nsmap:
            E = ElementMaker(namespace=ns, nsmap=nsmap)
        args = []
        if self.text is not None:
            args.append(self.text)
        for key_branches in self._branches.itervalues():
            args.extend(sb.to_xml(default_ns=None, E=E, **kwargs)
                        for sb in key_branches)
        return E(self.name, *args, **self._attr)

    @classmethod
    def from_xml(cls, element, **kwargs):  # @UnusedVariable
        name = strip_xmlns(element.tag)
        branches = defaultdict(list)
        for child in element.getchildren():
            branches[strip_xmlns(child.tag)].append(
                _AnnotationsBranch.from_xml(child))
        attr = dict(element.attrib)
        text = element.text if element.text else None
        return cls(name, attr, branches, text=text)

    def _copy_to_clone(self, clone, memo, **kwargs):
        self._clone_defining_attr(clone, memo, **kwargs)


VALIDATION = 'Validation'
DIMENSIONALITY = 'dimensionality'
PY9ML_NS = 'http://github.com/INCF/lib9ml'

xml_visitor_module_re = re.compile(r'nineml\.abstraction\.\w+\.visitors\.xml')


from nineml.utils import expect_none_or_single  # @IgnorePep8
