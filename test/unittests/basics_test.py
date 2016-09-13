import unittest
import collections
from nineml.utils.testing.comprehensive import (
    all_types, instances_of_all_types)
from nineml.abstraction.ports import SendPortBase


class TestAccessors(unittest.TestCase):

    exceptions = [('AnalogReceivePortExposure', '_name'),
                  ('AnalogSendPortExposure', '_name'),
                  ('AnalogReducePortExposure', '_name'),
                  ('EventReceivePortExposure', '_name'),
                  ('EventSendPortExposure', '_name'),
                  ('EventPortConnection', '_sender_role'),
                  ('AnalogPortConnection', '_sender_role'),
                  ('EventPortConnection', '_receiver_role'),
                  ('AnalogPortConnection', '_receiver_role'),
                  ('EventPortConnection', '_sender_name'),
                  ('AnalogPortConnection', '_sender_name'),
                  ('EventPortConnection', '_receiver_name'),
                  ('AnalogPortConnection', '_receiver_name')]

    def test_defining_attributes(self):
        for name, cls in all_types.iteritems():
            if hasattr(cls, 'defining_attributes'):
                for attr_name in getattr(cls, 'defining_attributes'):
                    for elem in instances_of_all_types[name]:
                        if (not isinstance(elem, basestring) and
                                isinstance(elem, collections.Iterable)):
                            continue
                        attr = getattr(elem, attr_name)
                        if (attr_name.startswith('_') and
                            hasattr(cls, attr_name[1:]) and
                                (name, attr_name) not in self.exceptions):
                            self.assertEqual(attr,
                                             getattr(elem, attr_name[1:]))

    def test_member_accessors(self):
        """
        Each "ContainerObject" provides accessor methods for each member type
        it contains that return:
            - iterator over each member
            - iterator over the names of the members
            - accessor for an individual member using the name
            - number of members

        This test checks to see whether they are internally consistent and of
        the right type
        """
        for name, cls in all_types.iteritems():
            if hasattr(cls, 'class_to_member'):
                for elem in instances_of_all_types[name]:
                    for member in cls.class_to_member:
                        num = elem._num_members(member, cls.class_to_member)
                        names = list(elem._member_names_iter(
                            member, cls.class_to_member))
                        members = sorted(elem._members_iter(
                            member, cls.class_to_member))
                        accessor_members = sorted(
                            elem._member_accessor(member,
                                                  cls.class_to_member)(n)
                            for n in names)
                        # Check num_* matches number of members and names
                        self.assertIsInstance(
                            num, int, ("num_{} did not return an integer ({})"
                                       .format(cls.class_to_member[member],
                                               num)))
                        self.assertEqual(
                            len(members), num,
                            "num_{} did not return the same length ({}) as the"
                            " number of members ({})".format(
                                cls.class_to_member[member], num,
                                len(members)))
                        self.assertEqual(
                            len(names), num,
                            "num_{} did not return the same length ({}) as the"
                            " number of names ({})".format(
                                cls.class_to_member[member], num,
                                len(names)))
                        # Check all names are strings and don't contain
                        # duplicates
                        self.assertTrue(
                            all(isinstance(n, basestring) for n in names),
                            "Not all names of {} in '{} {} were strings "
                            "('{}')".format(member, elem._name, name,
                                            names))
                        self.assertEqual(
                            len(names), len(set(names)),
                            "Duplicate names found in {} members of '{}' {} "
                            "('{}')".format(member, elem._name, name, names))
                        # Check all members are of the correct type
                        self.assertTrue(
                            all(m.nineml_type == member for m in members),
                            "Not all {} members accessed in '{}' {} via "
                            "iterator were of {} type ({})"
                            .format(member, elem._name, name, member,
                                    ', '.join(str(m) for m in members)))
                        self.assertEqual(
                            members, accessor_members,
                            "{} members accessed through iterator ({}) do not "
                            "match members accessed through individual "
                            "accessor method ({}) for '{}' {}"
                            .format(member, members, accessor_members,
                                    elem._name, name))
                    total_num = elem.num_elements()
                    all_names = list(elem.element_names())
                    all_members = sorted(elem.elements())
                    all_accessor_members = sorted(
                        elem.element(n, include_send_ports=True)
                        for n in all_names)
                    self.assertIsInstance(
                        total_num, int,
                        ("num_elements did not return an integer ({})"
                         .format(total_num)))
                    self.assertEqual(
                        len(all_members), total_num,
                        "num_elements did not return the same length "
                        "({}) as the number of all_members ({})".format(
                            total_num, len(all_members)))
                    # Check all all_names are strings and don't contain
                    # duplicates
                    self.assertTrue(
                        all(isinstance(n, basestring) for n in all_names),
                        "Not all element names in '{} {} were strings "
                        "('{}')".format(elem._name, name, all_names))
                    self.assertEqual(
                        len(all_names), len(set(all_names)),
                        "Duplicate element names found in '{}' {} "
                        "('{}')".format(elem._name, name, all_names))
                    # Check all all_members are of the correct type
                    self.assertGreaterEqual(
                        len(all_members), len(all_names),
                        "The length of all members ({}) should be equal to or "
                        "greater than the length of member names ({}), which "
                        "wasn't the case for '{}' {}. NB: "
                        "send ports will be masked by state variables and "
                        "aliases".format(len(all_members), len(all_names),
                                         elem._name, name))
                    diff = set(all_members) - set(all_accessor_members)
                    self.assertTrue(
                        all(isinstance(m, SendPortBase) for m in diff),
                        "Elements accessed through iterator ({}) do not "
                        "match all_members accessed through individual "
                        "accessor method ({}) for '{}' {}, with the exception "
                        "of send ports"
                        .format(all_members, all_accessor_members,
                                elem._name, name))

    def test_port_accessors(self):
        for cls_name in ('Dynamics', 'DynamicsProperties', 'MultiDynamics',
                         'MultiDynamicsProperties', 'Population', 'Selection'):
            cls = all_types[cls_name]
            for elem in instances_of_all_types[cls_name]:
                for prefix in ('', 'receive_', 'send_', 'analog_', 'event_',
                               'analog_receive_', 'event_receive_',
                               'analog_reduce_'):
                    num = getattr(elem, 'num_{}ports'.format(prefix))
                    names = list(getattr(elem, '{}port_names'.format(prefix)))
                    members = sorted(getattr(elem, '{}ports'.format(prefix)))
                    accessor_members = sorted(
                        getattr(elem, '{}port'.format(prefix))(n)
                        for n in names)
                    # Check num_* matches number of members and names
                    self.assertEqual(
                        len(members), num,
                        "num_{}ports did not return the same length ({}) as "
                        "the number of members ({})".format(prefix, num,
                                                            len(members)))
                    self.assertEqual(
                        len(names), num,
                        "num_{}ports did not return the same length ({}) as "
                        "the number of names ({})".format(prefix, num,
                                                          len(names)))
                    # Check all names are strings and don't contain
                    # duplicates
                    self.assertEqual(
                        len(names), len(set(names)),
                        "Duplicate names found in {}ports of '{}' {} "
                        "('{}')".format(prefix, elem._name, cls_name, names))
                    self.assertEqual(
                        members, accessor_members,
                        "{}ports accessed through iterator ({}) do not "
                        "match members accessed through individual "
                        "accessor method ({}) for '{}' {}"
                        .format(prefix, members, accessor_members,
                                elem._name, cls_name))


class TestRepr(unittest.TestCase):

    def test_repr(self):
        for name, elems in instances_of_all_types.iteritems():
            for elem in elems:
                if name == 'NineMLDocument':
                    self.assertTrue(repr(elem).startswith('Document'))
                else:
                    self.assertTrue(
                        repr(elem).startswith(name),
                        "__repr__ for {} instance does not start with '{}' "
                        "('{}')"
                        .format(name, all_types[name].__name__, repr(elem)))