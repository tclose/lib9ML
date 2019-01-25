"""
This file contains utility classes for modifying components.

:copyright: Copyright 2010-2017 by the NineML Python team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from collections import defaultdict
import sympy
from .base import BaseDynamicsVisitor
from ...componentclass.visitors.modifiers import (
    ComponentRenameSymbol, ComponentSubstituteAliases)


class DynamicsRenameSymbol(ComponentRenameSymbol,
                           BaseDynamicsVisitor):

    """ Can be used for:
    StateVariables, Aliases, Ports
    """

    def action_dynamics(self, dynamics, **kwargs):
        return self.action_componentclass(dynamics, **kwargs)

    def action_regime(self, regime, **kwargs):  # @UnusedVariable @IgnorePep8
        if regime.name == self.old_symbol_name:
            regime._name = self.new_symbol_name
        regime._update_member_key(
            self.old_symbol_name, self.new_symbol_name)
        # Update the on condition trigger keys, which can't be updated via
        # the _update_member_key method
        for trigger in list(regime.on_condition_triggers):
            if sympy.Symbol(self.old_symbol_name) in trigger.free_symbols:
                new_trigger = trigger.xreplace(
                    {sympy.Symbol(self.old_symbol_name):
                     sympy.Symbol(self.new_symbol_name)})
                regime._on_conditions[new_trigger] = (regime._on_conditions.
                                                      pop(trigger))

    def action_statevariable(self, state_variable, **kwargs):  # @UnusedVariable @IgnorePep8
        if state_variable.name == self.old_symbol_name:
            state_variable._name = self.new_symbol_name
            self.note_lhs_changed(state_variable)

    def action_analogsendport(self, port, **kwargs):  # @UnusedVariable
        self._action_port(port, **kwargs)

    def action_analogreceiveport(self, port, **kwargs):  # @UnusedVariable
        self._action_port(port, **kwargs)

    def action_analogreduceport(self, port, **kwargs):  # @UnusedVariable
        self._action_port(port, **kwargs)

    def action_eventsendport(self, port, **kwargs):  # @UnusedVariable
        self._action_port(port, **kwargs)

    def action_eventreceiveport(self, port, **kwargs):  # @UnusedVariable
        self._action_port(port, **kwargs)

    def action_outputevent(self, event_out, **kwargs):  # @UnusedVariable
        if event_out.port_name == self.old_symbol_name:
            event_out._port_name = self.new_symbol_name
            self.note_rhs_changed(event_out)

    def action_stateassignment(self, assignment, **kwargs):  # @UnusedVariable
        if self.old_symbol_name in assignment.atoms:
            self.note_rhs_changed(assignment)
            assignment.name_transform_inplace(self.namemap)

    def action_timederivative(self, timederivative, **kwargs):  # @UnusedVariable @IgnorePep8
        if timederivative.variable == self.old_symbol_name:
            self.note_lhs_changed(timederivative)
            timederivative.name_transform_inplace(self.namemap)
        elif self.old_symbol_name in timederivative.atoms:
            self.note_rhs_changed(timederivative)
            timederivative.name_transform_inplace(self.namemap)

    def action_trigger(self, trigger, **kwargs):  # @UnusedVariable
        if self.old_symbol_name in trigger.rhs_atoms:
            self.note_rhs_changed(trigger)
            trigger.rhs_name_transform_inplace(self.namemap)

    def action_oncondition(self, on_condition, **kwargs):  # @UnusedVariable
        if on_condition._target_regime == self.old_symbol_name:
            on_condition._target_regime = self.new_symbol_name
        on_condition._update_member_key(
            self.old_symbol_name, self.new_symbol_name)

    def action_onevent(self, on_event, **kwargs):  # @UnusedVariable
        if on_event.src_port_name == self.old_symbol_name:
            on_event._src_port_name = self.new_symbol_name
            self.note_rhs_changed(on_event)
        if on_event._target_regime.name == self.old_symbol_name:
            on_event._target_regime._name = self.new_symbol_name
        on_event._update_member_key(
            self.old_symbol_name, self.new_symbol_name)


class DynamicsSubstituteAliases(ComponentSubstituteAliases,
                                BaseDynamicsVisitor):

    def action_dynamics(self, dynamics, **kwargs):  # @UnusedVariable
        self.outputs.update(dynamics.analog_send_port_names)

    def action_timederivative(self, time_derivative, **kwargs):  # @UnusedVariable @IgnorePep8
        self.substitute(time_derivative)

    def action_stateassignment(self, state_assignment, **kwargs):  # @UnusedVariable @IgnorePep8
        self.substitute(state_assignment)

    def action_trigger(self, trigger, **kwargs):  # @UnusedVariable @IgnorePep8
        old_rhs = trigger.rhs
        self.substitute(trigger)
        # If trigger expression has changed update on_condition key
        if trigger.rhs != old_rhs:
            member_dict = self.contexts[-2].parent._on_conditions
            member_dict[trigger.rhs] = member_dict.pop(old_rhs)

    def post_action_dynamics(self, dynamics, results, **kwargs):  # @UnusedVariable @IgnorePep8
        self.remove_uneeded_aliases(dynamics)

    def post_action_regime(self, regime, results, **kwargs):  # @UnusedVariable
        self.remove_uneeded_aliases(regime)


class DynamicsMergeLinearSubComponents(BaseDynamicsVisitor):
    """
    Attempts to combine sub-components with equivalent linear dynamics in
    order to reduce the number of state-variables in the multi-dynamics
    """

    def __init__(self, multi_dynamics, multi_properties=None):
        self.multi_dynamics = self.multi_dynamics
        # Flatten multi dynamics so we can remove unrequired states + dynamics
        self.merged = multi_dynamics.flatten()

    def _group_properties(self, multi_properties):
        for comp in multi_properties.sub_components:
            matching_td_props = defaultdict(list)
            # Sort components into matching properties used in ODES
            for node in nodes:
                props = node['properties']
                td_props = frozenset(
                    props[p] for p in self.time_derivative_parameters)
                matching_td_props[td_props].append(props)
