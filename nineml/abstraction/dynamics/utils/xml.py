"""
docstring needed

:copyright: Copyright 2010-2013 by the Python lib9ML team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from itertools import chain
from nineml.annotations import annotate_xml
from nineml.utils import expect_single
from nineml.xmlns import E
from ..base import DynamicsClass
from nineml.annotations import read_annotations
from ...ports import (EventSendPort, EventReceivePort, AnalogSendPort,
                      AnalogReceivePort, AnalogReducePort)
from ..transitions import (
    OnEvent, OnCondition, StateAssignment, OutputEvent, Trigger)
from ..regimes import Regime, StateVariable, TimeDerivative
from ...componentclass.utils.xml import (
    ComponentClassXMLLoader, ComponentClassXMLWriter)
from nineml.exceptions import handle_xml_exceptions


class DynamicsXMLLoader(ComponentClassXMLLoader):

    """This class is used by XMLReader interny.

    This class loads a NineML XML tree, and stores
    the components in ``components``. It o records which file each XML node
    was loaded in from, and stores this in ``component_srcs``.

    """

    @read_annotations
    @handle_xml_exceptions
    def load_componentclass(self, element):

        block_names = ('Parameter', 'AnalogSendPort', 'AnalogReceivePort',
                       'EventSendPort', 'EventReceivePort', 'AnalogReducePort',
                       'Dynamics', 'Regime', 'Alias', 'StateVariable',
                       'Constant')

        blocks = self._load_blocks(element, blocks=block_names)

        return Dynamics(
            name=element.attrib['name'],
            parameters=blocks["Parameter"],
            analog_ports=chain(blocks["AnalogSendPort"],
                               blocks["AnalogReceivePort"],
                               blocks["AnalogReducePort"]),
            event_ports=chain(blocks["EventSendPort"],
                              blocks["EventReceivePort"]),
            regimes=blocks["Regime"],
            aliases=blocks["Alias"],
            state_variables=blocks["StateVariable"],
            constants=blocks["Constant"],
            url=self.document.url)

    @read_annotations
    @handle_xml_exceptions
    def load_eventsendport(self, element):
        return EventSendPort(name=element.attrib['name'])

    @read_annotations
    @handle_xml_exceptions
    def load_eventreceiveport(self, element):
        return EventReceivePort(name=element.attrib['name'])

    @read_annotations
    @handle_xml_exceptions
    def load_analogsendport(self, element):
        return AnalogSendPort(
            name=element.attrib['name'],
            dimension=self.document[element.attrib['dimension']])

    @read_annotations
    @handle_xml_exceptions
    def load_analogreceiveport(self, element):
        return AnalogReceivePort(
            name=element.attrib['name'],
            dimension=self.document[element.attrib['dimension']])

    @read_annotations
    @handle_xml_exceptions
    def load_analogreduceport(self, element):
        return AnalogReducePort(
            name=element.attrib['name'],
            dimension=self.document[element.attrib['dimension']],
            operator=element.attrib['operator'])

    @read_annotations
    @handle_xml_exceptions
    @handle_xml_exceptions
    def load_regime(self, element):
        block_names = ('TimeDerivative', 'OnCondition', 'OnEvent')
        blocks = self._load_blocks(element, blocks=block_names)
        transitions = blocks["OnEvent"] + blocks['OnCondition']
        return Regime(name=element.attrib['name'],
                      time_derivatives=blocks["TimeDerivative"],
                      transitions=transitions)

    @read_annotations
    @handle_xml_exceptions
    def load_statevariable(self, element):
        name = element.attrib['name']
        dimension = self.document[element.attrib['dimension']]
        return StateVariable(name=name, dimension=dimension)

    @read_annotations
    @handle_xml_exceptions
    def load_timederivative(self, element):
        variable = element.attrib['variable']
        expr = self.load_single_internmaths_block(element)
        return TimeDerivative(variable=variable,
                              rhs=expr)

    @read_annotations
    @handle_xml_exceptions
    def load_oncondition(self, element):
        block_names = ('Trigger', 'StateAssignment', 'OutputEvent')
        blocks = self._load_blocks(element, blocks=block_names)
        target_regime = element.attrib['target_regime']
        trigger = expect_single(blocks["Trigger"])
        return OnCondition(trigger=trigger,
                           state_assignments=blocks["StateAssignment"],
                           output_events=blocks["OutputEvent"],
                           target_regime=target_regime)

    @read_annotations
    @handle_xml_exceptions
    def load_onevent(self, element):
        block_names = ('StateAssignment', 'OutputEvent')
        blocks = self._load_blocks(element, blocks=block_names)
        target_regime = element.attrib['target_regime']
        return OnEvent(src_port_name=element.attrib['port'],
                       state_assignments=blocks["StateAssignment"],
                       output_events=blocks["OutputEvent"],
                       target_regime=target_regime)

    def load_trigger(self, element):
        return Trigger(self.load_single_internmaths_block(element))

    @read_annotations
    @handle_xml_exceptions
    def load_stateassignment(self, element):
        lhs = element.attrib['variable']
        rhs = self.load_single_internmaths_block(element)
        return StateAssignment(lhs=lhs, rhs=rhs)

    @read_annotations
    @handle_xml_exceptions
    def load_outputevent(self, element):
        port_name = element.attrib['port']
        return OutputEvent(port_name=port_name)

    tag_to_loader = {
        "ComponentClass": load_componentclass,
        "Regime": load_regime,
        "StateVariable": load_statevariable,
        "EventSendPort": load_eventsendport,
        "AnalogSendPort": load_analogsendport,
        "EventReceivePort": load_eventreceiveport,
        "AnalogReceivePort": load_analogreceiveport,
        "AnalogReducePort": load_analogreduceport,
        "OnCondition": load_oncondition,
        "OnEvent": load_onevent,
        "TimeDerivative": load_timederivative,
        "Trigger": load_trigger,
        "StateAssignment": load_stateassignment,
        "OutputEvent": load_outputevent,
    }


class DynamicsXMLWriter(ComponentClassXMLWriter):

    @annotate_xml
    def visit_componentclass(self, componentclass):
        return E('ComponentClass',
                 *[e.accept_visitor(self) for e in componentclass],
                 name=componentclass.name)

    @annotate_xml
    def visit_regime(self, regime):
        return E('Regime', name=regime.name,
                 *[e.accept_visitor(self) for e in regime])

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
                 dimension=port.dimension.name, operator=port.operator)

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
    def visit_stateassignment(self, assignment):
        return E('StateAssignment',
                 E("MathInline", assignment.rhs_cstr),
                 variable=assignment.lhs)

    @annotate_xml
    def visit_timederivative(self, time_derivative):
        return E('TimeDerivative',
                 E("MathInline", time_derivative.rhs_cstr),
                 variable=time_derivative.variable)

    @annotate_xml
    def visit_oncondition(self, on_condition):
        return E('OnCondition', on_condition.trigger.accept_visitor(self),
                 target_regime=on_condition._target_regime.name,
                 *[e.accept_visitor(self) for e in on_condition])

    @annotate_xml
    def visit_trigger(self, trigger):
        return E('Trigger', E("MathInline", trigger.rhs_cstr))

    @annotate_xml
    def visit_onevent(self, on_event):
        return E('OnEvent', port=on_event.src_port_name,
                 target_regime=on_event.target_regime.name,
                 *[e.accept_visitor(self) for e in on_event])