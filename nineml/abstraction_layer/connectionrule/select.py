"""
This file contains the definitions for the Events

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""

from nineml.utils import ensure_valid_identifier
from nineml.abstraction_layer.componentclass import BaseALObject
from ..expressions import Expression, RandomVariable
from .utils.cloner import ConnectionRuleCloner
from nineml.base import MemberContainerObject
from nineml.utils import normalise_parameter_as_list, assert_no_duplicates
from .utils.visitors import ConnectionRuleElementFinder


class Number(BaseALObject, Expression):

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_number(self, **kwargs)

    def __init__(self, rhs):
        BaseALObject.__init__(self)
        Expression.__init__(self, rhs)

    def __repr__(self):
        return "Number('%s')" % (self.rhs)


class Mask(BaseALObject, Expression):

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_mask(self, **kwargs)

    def __init__(self, rhs):
        BaseALObject.__init__(self)
        Expression.__init__(self, rhs)

    def __repr__(self):
        return "Mask('%s')" % (self.rhs)


class Preference(BaseALObject, Expression):

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_preference(self, **kwargs)

    def __init__(self, rhs):
        BaseALObject.__init__(self)
        Expression.__init__(self, rhs)

    def __repr__(self):
        return "Preference('%s')" % (self.rhs)


class RepeatWhile(BaseALObject, Expression):

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_repeatwhile(self, **kwargs)

    def __init__(self, rhs, stages=1):
        BaseALObject.__init__(self)
        Expression.__init__(self, rhs)
        assert isinstance(stages, int)
        self._stages = stages

    @property
    def stages(self):
        return self._stages

    def __repr__(self):
        return "RepeatWhile('{}', stages {})".format(self.rhs, self.stages)


class Selected(BaseALObject):

    """Selected
    """

    defining_attributes = ('_name', '_scope')

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_selected(self, **kwargs)

    def __init__(self, name, scope='all'):
        """OutputEvent Constructor

        :param port: The name of the output EventPort that should
            transmit an event. An `EventPort` with a mode of 'send' must exist
            with a corresponding name in the component_class, otherwise a
            ``NineMLRuntimeException`` will be raised.

        """
        super(Selected, self).__init__()
        self._name = name.strip()
        self._scope = scope
        ensure_valid_identifier(self._name)

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    def __str__(self):
        return 'Selected(name: {}, scope: {} )'.format(self.name,
                                                           self.scope)

    def __repr__(self):
        return "Selected(name='{}', scope='{}')".format(self.name,
                                                           self.scope)


class NumberSelected(BaseALObject):

    """NumberSelected
    """

    defining_attributes = ('_name', '_scope', '_perspective')

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_perspective(self, **kwargs)

    def __init__(self, name, scope='all', perspective='all'):
        """OutputEvent Constructor

        :param port: The name of the output EventPort that should
            transmit an event. An `EventPort` with a mode of 'send' must exist
            with a corresponding name in the component_class, otherwise a
            ``NineMLRuntimeException`` will be raised.

        """
        super(NumberSelected, self).__init__()
        self._name = name.strip()
        self._scope = scope
        self._perspective = perspective
        ensure_valid_identifier(self._name)

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def perspective(self):
        return self._perspective

    def __str__(self):
        return 'NumberSelected( name: {}, scope: {} )'.format(self.name,
                                                              self.scope)

    def __repr__(self):
        return "NumberSelected(name='{}', scope='{}')".format(self.name,
                                                              self.scope)


class Select(BaseALObject, MemberContainerObject):

    defining_attributes = ('_mask', '_number', '_preference',
                           '_was_selecteds', '_number_selecteds',
                           '_random_variables', '_select',
                           '_repeat_while')
    class_to_member_dict = {Selected: '_was_selecteds',
                            NumberSelected: '_number_selecteds',
                            RandomVariable: '_random_variables',
                            RepeatWhile: '_repeat_whiles'}

    def __init__(self, mask=None, number=None, preference=None,
                 was_selecteds=None, number_selecteds=None,
                 random_variables=None, select=None, repeat_whiles=None):
        """Abstract class representing a transition from one |Regime| to
        another.

        |Transition| objects are not created directly, but via the subclasses
        |OnEvent| and |OnCondition|.

        :param was_selecteds: A list of the state-assignments performed
            when this transition occurs. Objects in this list are either
            `string` (e.g A = A+13) or |Selected| objects.
        :param number_selecteds: A list of |NumberSelected| objects emitted
            when this transition occurs.
        :param target_regime_name: The name of the regime to go into after this
            transition.  ``None`` implies staying in the same regime. This has
            to be specified as a string, not the object, because in general the
            |Regime| object is not yet constructed. This is automatically
            resolved by the |ConnectionRule| in
            ``_ResolveTransitionRegimeNames()`` during construction.


        .. todo::

            For more information about what happens at a regime transition, see
            here: XXXXXXX

        """
        BaseALObject.__init__(self)
        MemberContainerObject.__init__(self)

        self._mask = mask
        self._number = number
        self._preference = preference
        self._select = select

        # Load state-assignment objects as strings or Selected objects
        was_selecteds = normalise_parameter_as_list(was_selecteds)
        number_selecteds = normalise_parameter_as_list(number_selecteds)
        random_variables = normalise_parameter_as_list(random_variables)
        repeat_whiles = normalise_parameter_as_list(repeat_whiles)

        assert_no_duplicates(s.name for s in was_selecteds)
        assert_no_duplicates(ns.name for ns in number_selecteds)
        assert_no_duplicates(rv.name for rv in random_variables)
        assert_no_duplicates(rw.level for rw in repeat_whiles)

        self._was_selecteds = dict((s.name, s) for s in was_selecteds)
        self._number_selecteds = dict((ns.name, ns) for ns in number_selecteds)
        self._random_variables = dict((rv.name, rv) for rv in random_variables)
        self._repeat_whiles = dict((rw.stages, rw) for rw in repeat_whiles)

    def __copy__(self):
        return ConnectionRuleCloner(self)

    def _find_element(self, element):
        return ConnectionRuleElementFinder(element).found_in(self)

    @property
    def number(self):
        return self._number

    @property
    def mask(self):
        return self._mask

    @property
    def preference(self):
        return self._preference

    @property
    def select(self):
        return self._select

    @property
    def was_selecteds(self):
        return self._was_selecteds.itervalues()

    def was_selected(self, name):
        return self._was_selecteds[name]

    @property
    def was_selected_names(self):
        return self._was_selecteds.iterkeys()

    @property
    def number_selecteds(self):
        return self._number_selecteds.itervalues()

    def number_selected(self, name):
        return self._number_selecteds[name]

    @property
    def number_selected_names(self):
        return self._number_selecteds.iterkeys()

    @property
    def random_variables(self):
        return self._random_variables.itervalues()

    @property
    def random_variable_names(self):
        return self._random_variables.iterkeys()

    @property
    def random_variable(self, name):
        return self._random_variables[name]

    @property
    def repeat_whiles(self):
        return self._repeat_whiles.itervalues()

    @property
    def repeat_while_stages(self):
        return self._repeat_whiles.iterkeys()

    @property
    def repeat_while(self, name):
        return self._repeat_whiles[name]
