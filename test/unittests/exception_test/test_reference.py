import unittest
from nineml.reference import (BaseReference)
from nineml.utils.testing.comprehensive import instances_of_all_types
from nineml.exceptions import (NineMLXMLAttributeError, NineMLRuntimeError)


class TestBaseReferenceExceptions(unittest.TestCase):

    def test___init___ninemlruntimeerror(self):
        """
        line #: 25
        message: Must supply a document with a non-None URL that is being referenced from if definition is a relative URL string, '{}'

        context:
        --------
    def __init__(self, name, document, url=None):
        super(BaseReference, self).__init__()
        if url:
            if url.startswith('.'):
                if document is None or document.url is None:
        """

        basereference = instances_of_all_types['BaseReference']
        self.assertRaises(
            NineMLRuntimeError,
            basereference.__init__,
            name=None,
            document=None,
            url=None)

    def test_from_xml_ninemlxmlattributeerror(self):
        """
        line #: 81
        message: References require the element name provided in the XML element text

        context:
        --------
    def from_xml(cls, element, document, **kwargs):  # @UnusedVariable
        xmlns = extract_xmlns(element.tag)
        if xmlns == NINEMLv1:
            name = element.text
            if name is None:
        """

        self.assertRaises(
            NineMLXMLAttributeError,
            BaseReference.from_xml,
            element=None,
            document=None)

