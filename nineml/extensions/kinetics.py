"""
Description of the module goes here

Copyright info here

Author: Russell Jarvis
Date: 27/2/2015
"""
from nineml.abstraction_layer.dynamics import DynamicsBlock
from nineml.abstraction_layer import BaseALObject
from nineml.abstraction_layer.expressions import (
    Expression, Alias, ExpressionSymbol)
from nineml.abstraction_layer.dynamics.utils.xml import (
    DynamicsClassXMLLoader, DynamicsClassXMLWriter)
from nineml.annotations import read_annotations, annotate_xml
from nineml.abstraction_layer.units import dimensionless
from nineml.utils import ensure_valid_identifier, expect_single
from nineml.exceptions import NineMLRuntimeError
from nineml.utils import normalise_parameter_as_list
from nineml.abstraction_layer.dynamics import DynamicsClass
from nineml.utils import check_list_contain_same_items
from itertools import chain
from nineml.abstraction_layer.dynamics import (
    TimeDerivative, Regime, StateVariable)
import sympy
import nineml
from nineml.xmlns import E

def inf_check(l1, l2, desc):
    check_list_contain_same_items(l1, l2, desc1='Declared',
                                  desc2='Inferred', ignore=['t'], desc=desc)


class KineticsClass(DynamicsClass):
    defining_attributes = ('_name', '_parameters', '_analog_send_ports',
                           '_analog_receive_ports', '_analog_reduce_ports',
                           '_event_send_ports', '_event_receive_ports',
                           '_kineticsblock', '_kinetic_states',
                           '_reactions', '_constraints', '_main_block')

    def __init__(self, name,
                 parameters=None, analog_ports=[],
                 event_ports=[], aliases=None, constants=None,
                 kinetic_states=None, reactions=None, constraints=None,
                 kineticsblock=None):

        # We can specify in the componentclass, and they will get forwarded to
        # the dynamics class. We check that we do not specify half-and-half:
        if kineticsblock is not None:
            if (constraints is not None or reactions is not None
                    or kinetic_states is not None or aliases is not None):
                raise NineMLRuntimeError(
                    "Either specify a 'kineticdynamics' parameter, or "
                    "kinetic_states, reactions,constraints, but not both!")
        else:
            kineticsblock = KineticsBlock(
                kinetic_states=kinetic_states, reactions=reactions,
                constraints=constraints, constants=constants, aliases=aliases)

        super(KineticsClass, self).__init__(
            name=name, parameters=parameters, event_ports=event_ports,
            analog_ports=analog_ports, dynamicsblock=kineticsblock)

    @property
    def reactions(self):
        return self._main_block.reactions.itervalues()

    def reaction(self, from_state, to_state):
        return self._main_block.reactions[tuple(sorted((from_state,
                                                        to_state)))]

    @property
    def kinetic_states(self):
        return self._main_block.kinetic_states.itervalues()

    def kinetic_state(self, name):
        return self._main_block.kinetic_states[name]

    @property
    def kinetic_state_names(self):
        return self._main_block.kinetic_states.iterkeys()

    @property
    def constraints(self):
        return self._main_block.constraints

    def constraint(self, name):
        return self._main_block.constraints[name]

    @property
    def constraint_variables(self):
        return self._main_block.constraints.iterkeys()

    @annotate_xml
    def to_xml(self):
        self.standardize_unit_dimensions()
        XMLWriter=nineml.extensions.kinetics.KineticsClassXMLWriter
        self.validate()
        return XMLWriter().visit(self)


class KineticsBlock(DynamicsBlock):

    """
    An object, which encapsulates a component's regimes, transitions,
    and state variables
    """

    defining_attributes = ('kinetic_states', 'aliases', 'reactions',
                           'constraints', 'constants')

    def __init__(self, kinetic_states=None, reactions=None, constraints=None,
                 constants=None, aliases=None):
        """DynamicsBlock object constructor

           :param aliases: A list of aliases, which must be either |Alias|
               objects or ``string``s.
           :param regimes: A list containing at least one |Regime| object.
           :param state_variables: An optional list of the state variables,
                which can either be |StateVariable| objects or `string` s. If
                provided, it must match the inferred state-variables from the
                regimes; if it is not provided it will be inferred
                automatically.
        """

        aliases = normalise_parameter_as_list(aliases)
        kinetic_states = normalise_parameter_as_list(kinetic_states)
        reactions = normalise_parameter_as_list(reactions)
        constraints = normalise_parameter_as_list(constraints)

        # Kinetics specific members
        self.reactions = dict((tuple(sorted((r.from_state, r.to_state))), r)
                               for r in reactions)
  
        
        self.kinetic_states = dict((a.name, a) for a in kinetic_states)


        if type(constraints[0])==tuple:
            state=kinetic_states[0]
            self.constraints = dict((state, c) for c in constraints)
        else:
            self.constraints = dict((c.state, c) for c in constraints)

        td_rates = {}
        for state in self.kinetic_states:
            td_rates[state] = ([], [])
        for reaction in self.reactions.itervalues():
            td_rates[reaction.from_state][0].append(reaction.forward_rate)
            td_rates[reaction.from_state][1].append((reaction.reverse_rate,
                                                     reaction.to_state))
            td_rates[reaction.to_state][0].append(reaction.reverse_rate)
            td_rates[reaction.to_state][1].append((reaction.forward_rate,
                                                   reaction.from_state))

        # Create all of the time_derivative objects. Append them to a list.
        time_derivatives = []
        for state_name, (outgoing_rates,
                         incoming_rates) in td_rates.iteritems():
            td = TimeDerivative(dependent_variable=state_name, rhs='0')
            if (state_name not in (c.state for c in constraints)):
                for rate in outgoing_rates:
                    outcoming_state = self.kinetic_states[state_name]
                    td -= rate * outcoming_state

                    # print type(td), type(rate), type(outcoming_state)
                    # state_subs
                for rate, incoming_state_name in incoming_rates:
                    incoming_state = self.kinetic_states[incoming_state_name]
                    td += rate * incoming_state
                time_derivatives.append(td)

        # Construct base class members and pass to base class __init__

        # Aliases are also referencing state
        for reaction in reactions:
            aliases.append(reaction.forward_rate)
            aliases.append(reaction.reverse_rate)

        for ceq in self.constraints.itervalues():
            reduce_state = self.kinetic_states[ceq.state]
            state_subs = -(ceq.rhs - reduce_state)

            state_subs.subs(reduce_state, state_subs)
            for td in time_derivatives:
                td.rhs = td.rhs.subs(reduce_state, state_subs)

        regimes = [Regime(name="default", time_derivatives=time_derivatives)]
        state_variables = [StateVariable(name=ks.name, dimension=dimensionless)
                           for ks in self.kinetic_states.itervalues()
                           if ks.name not in (
                               c.state for c in self.constraints.itervalues())]

        super(KineticsBlock, self).__init__(
            regimes=regimes, aliases=aliases, state_variables=state_variables,
            constants=constants)

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_dynamicsblock(self, **kwargs)

    def __repr__(self):
        return ('KineticsBlock({} regimes, {} aliases, {} state_variables, '
                '{} reactions, kinetic_states {}, constraints {})'
                .format(len(list(self.regimes)), len(list(self.aliases)),
                        len(list(self.state_variables)),
                        len(list(self.reactions)),
                        len(list(self.kinetic_states)),
                        len(list(self.constraints))))


class Constraint(Expression, BaseALObject):
    """
    New Class, Constraint Inherits from Alias, which inherits from BaseAL.
    """

    defining_attributes = ('_state', '_rhs')

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_statevariable(self, **kwargs)

    # constructor for making Constraint object.
    def __init__(self, expr, state):
        BaseALObject.__init__(self)
        Expression.__init__(self, expr)
        self._state = state

    @property
    def state(self):
        return self._state


class ReactionRate(Alias):

    def __init__(self, expr):
        self.rhs = expr
        self._reaction = None

    def set_reaction(self, reaction):
        if self._reaction is not None:
            raise NineMLRuntimeError(
                "Rate '{}' already belongs to another reaction block"
                .format(repr(self)))
        self._reaction = reaction

    @property
    def lhs(self):
        return self.name


class Reaction(BaseALObject):

    """A class representing a state-variable in a ``DynamicsClass``.

    This was originally a string, but if we intend to support units in the
    future, wrapping in into its own object may make the transition easier
    """

    defining_attributes = ('_from_state', ' _to_state', 'from_', 'to',
                           'ForwardRate', 'ReverseRate')

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_statevariable(self, **kwargs)

    def __init__(self, from_state, to_state, forward_rate, reverse_rate):

        """StateVariable Constructor

        :param name:  The name of the state variable.
        """

        self._from_state = from_state
        self._to_state = to_state
        self._forward_rate = forward_rate
        self._reverse_rate = reverse_rate

        # Reaction extends the BaseALObject
        self._forward_rate.set_reaction(self)
        self._reverse_rate.set_reaction(self)

    @property
    def name(self):
        return (('reaction__from_{}_to{}').format(self._from_state,
                                                  self._to_state))

    @property
    def from_state(self):
        return self._from_state

    @property
    def to_state(self):
        return self._to_state

    @property
    def forward_rate(self):
        return self._forward_rate

    @property
    def reverse_rate(self):
        return self._reverse_rate

    def set_dimension(self, dimension):
        self._dimension = dimension

    def __repr__(self):
        return ("Reaction(from={}, to={})"
                .format(self.from_state, self.to_state))


class KineticState(BaseALObject, ExpressionSymbol):

    """A class representing a state-variable in a ``DynamicsClass``.

    This was originally a string, but if we intend to support units in the
    future, wrapping in into its own object may make the transition easier
    """

    defining_attributes = ('name', 'dimension')

    def accept_visitor(self, visitor, **kwargs):
        """ |VISITATION| """
        return visitor.visit_statevariable(self, **kwargs)

    def __init__(self, name, dimension=None):
        """StateVariable Constructor

        :param name:  The name of the state variable.
        """
        self._name = name.strip()
        self._dimension = dimension if dimension is not None else dimensionless
        ensure_valid_identifier(self._name)

    @property
    def name(self):
        return self._name

    @property
    def dimension(self):
        return self._dimension

    def set_dimension(self, dimension):
        self._dimension = dimension

    def __repr__(self):
        return ("KineticState({}{})"
                .format(self.name,
                        ', dimension={}'.format(self.dimension.name)))

    def _sympy_(self):
        return sympy.Symbol(self.name)


class ForwardRate(ReactionRate):

    """Represents a first-order, ordinary differential equation with respect to
    time.

    """
    defining_attributes = ('_rhs')

    element_name = 'ForwardRate'

    @property
    def name(self):
        return 'ReactionRate__from{}_to{}'.format(self._reaction.from_state,
                                                  self._reaction.to_state)


class ReverseRate(ReactionRate):

    """Represents a first-order, ordinary differential equation with respect to
    time.

    """
    defining_attributes = ('_rhs')

    element_name = 'ReverseRate'

    @property
    def name(self):
        return 'ReactionRate__from{}_to{}'.format(self._reaction.to_state,
                                                  self._reaction.from_state)


class KineticsClassXMLWriter(DynamicsClassXMLWriter):

    @annotate_xml
    def visit_componentclass(self, componentclass):
        elements = ([p.accept_visitor(self)
                     for p in componentclass.analog_ports] +
                    [p.accept_visitor(self)
                     for p in componentclass.event_ports] +
                    [p.accept_visitor(self)
                     for p in componentclass.parameters] +
                    [componentclass._main_block.accept_visitor(self)])
        return E('ComponentClass', *elements, name=componentclass.name)

    @annotate_xml
    def visit_dynamicsblock(self, dynamicsblock):
        elements = ([b.accept_visitor(self)
                     for b in dynamicsblock.state_variables] +
                    [r.accept_visitor(self) for r in dynamicsblock.regimes] +
                    [b.accept_visitor(self) for b in dynamicsblock.aliases] +
                    [c.accept_visitor(self) for c in dynamicsblock.constants] +
                    [c.accept_visitor(self) for c in dynamicsblock.random_variables] +
                    [c.accept_visitor(self) for c in dynamicsblock.piecewises])
        return E('Dynamics', *elements)

    @annotate_xml
    def visit_regime(self, regime):
        nodes = ([node.accept_visitor(self)
                  for node in regime.time_derivatives] +
                 [node.accept_visitor(self) for node in regime.on_events] +
                 [node.accept_visitor(self) for node in regime.on_conditions])
        return E('Regime', name=regime.name, *nodes)

    @annotate_xml
    def visit_statevariable(self, state_variable):
        return E('StateVariable',
                 name=state_variable.name,
                 dimension=state_variable.dimension.name)

    @annotate_xml
    def visit_outputevent(self, event_out):
        return E('OutputEvent',
                 port=event_out.port_name)

    @annotate_xml
    def visit_analogreceiveport(self, port):
        return E('AnalogReceivePort', name=port.name,
                 dimension=port.dimension.name)

    @annotate_xml
    def visit_analogreduceport(self, port):
        return E('AnalogReducePort', name=port.name,
                 dimension=port.dimension.name, operator=port.reduce_op)

    @annotate_xml
    def visit_analogsendport(self, port):
        return E('AnalogSendPort', name=port.name,
                 dimension=port.dimension.name)

    @annotate_xml
    def visit_eventsendport(self, port):
        return E('EventSendPort', name=port.name)

    @annotate_xml
    def visit_eventreceiveport(self, port):
        return E('EventReceivePort', name=port.name)

    @annotate_xml
    def visit_assignment(self, assignment):
        return E('StateAssignment',
                 E("MathInline", assignment.rhs_str),
                 variable=assignment.lhs)

    @annotate_xml
    def visit_timederivative(self, time_derivative):
        return E('TimeDerivative',
                 E("MathInline", time_derivative.rhs_str),
                 variable=time_derivative.dependent_variable)

    @annotate_xml
    def visit_oncondition(self, on_condition):
        nodes = chain(on_condition.state_assignments,
                      on_condition.event_outputs, [on_condition.trigger])
        newNodes = [n.accept_visitor(self) for n in nodes]
        return E('OnCondition', *newNodes,
                 target_regime=on_condition._target_regime.name)

    @annotate_xml
    def visit_trigger(self, trigger):
        return E('Trigger', E("MathInline", trigger.rhs_str))

    @annotate_xml
    def visit_onevent(self, on_event):
        elements = ([p.accept_visitor(self)
                     for p in on_event.state_assignments] +
                    [p.accept_visitor(self) for p in on_event.event_outputs])
        return E('OnEvent', *elements, port=on_event.src_port_name,
                 target_regime=on_event.target_regime.name)


class KineticsClassXMLLoader(DynamicsClassXMLLoader):

    """This class is used by XMLReader internally.

    This class loads a NineML XML tree, and stores
    the components in ``components``. It o records which file each XML node
    was loaded in from, and stores this in ``component_srcs``.

    """

    @read_annotations
    def load_componentclass(self, element):

        blocks = ('Parameter', 'AnalogSendPort', 'AnalogReceivePort',
                  'EventSendPort', 'EventReceivePort', 'AnalogReducePort',
                  'Kinetics')

        subnodes = self._load_blocks(element, blocks=blocks)
        kineticsblock = expect_single(subnodes["Kinetics"])

        return KineticsClass(
            name=element.get('name'),
            parameters=subnodes["Parameter"],
            analog_ports=chain(subnodes["AnalogSendPort"],
                               subnodes["AnalogReceivePort"],
                               subnodes["AnalogReducePort"]),
            event_ports=chain(subnodes["EventSendPort"],
                              subnodes["EventReceivePort"]),
            kineticsblock=kineticsblock)

    @read_annotations
    def load_kineticsblock(self, element):

        subblocks = ('KineticState', 'Reaction', 'Constraint',
                     'ForwardRate', 'ReverseRate', 'Alias')
        subnodes = self._load_blocks(element, blocks=subblocks)

        return KineticsBlock(
            kinetic_states=subnodes["KineticState"],
            aliases=subnodes["Alias"],
            reactions=subnodes["Reaction"],
            constraints=subnodes["Constraint"],
        )

    @read_annotations
    def load_kineticstate(self, element):
        name = element.get("name")
        dimension = self.document[element.get('dimension')]
        return KineticState(name=name, dimension=dimension)

    @read_annotations
    def load_constraint(self, element):
        state = element.get("state")
        expr = self.load_single_internmaths_block(element)

        return Constraint(expr, state)

    @read_annotations
    def load_reaction(self, element):
        from_state = element.get("from")
        to_state = element.get("to")
        subblocks = ('ForwardRate', 'ReverseRate')
        subnodes = self._load_blocks(element, blocks=subblocks)
        # Gets the rate from the XML tag of the same name.
        forward_rate = expect_single(subnodes['ForwardRate'])
        reverse_rate = expect_single(subnodes['ReverseRate'])
        return Reaction(from_state, to_state, forward_rate, reverse_rate)

    @read_annotations
    def load_forwardrate(self, element):
        expr = self.load_single_internmaths_block(element)
        return ForwardRate(expr)

    @read_annotations
    def load_reverserate(self, element):
        expr = self.load_single_internmaths_block(element)
        return ReverseRate(expr)

    tag_to_loader = {
        "Kinetics": load_kineticsblock,
        "Constraint": load_constraint,
        "Reaction": load_reaction,
        "KineticState": load_kineticstate,
        "AnalogReceivePort": DynamicsClassXMLLoader.load_analogreceiveport,
        "AnalogSendPort": DynamicsClassXMLLoader.load_analogsendport,
        "AnalogReducePort": DynamicsClassXMLLoader.load_analogreduceport,
        "ForwardRate": load_forwardrate,
        "ReverseRate": load_reverserate,
    }
