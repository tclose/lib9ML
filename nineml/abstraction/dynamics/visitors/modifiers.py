"""
This file contains utility classes for modifying components.

:copyright: Copyright 2010-2017 by the NineML Python team, see AUTHORS.
:license: BSD-3, see LICENSE for details.
"""
from collections import defaultdict
from itertools import chain
import sympy
from .base import BaseDynamicsVisitor
from ..transitions import StateAssignment, OnCondition
from ...expressions.named import Alias
from nineml.visitors.base import BaseVisitorWithContext
from ...componentclass.visitors.modifiers import (
    ComponentRenameSymbol, ComponentSubstituteAliases)
from nineml.exceptions import NineMLNameError, NineMLCannotMergeException


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


class DynamicsMergeStatesOfLinearSubComponents(BaseVisitorWithContext,
                                               BaseDynamicsVisitor):
    """
    Attempts to combine the states of linear sub-components of the same class
    (typically synapses) in order to reduce the number of state-variables in
    the multi-dynamics class (and hence speed up the simulation).

    Parameters
    ----------
    multi_properties : MultiDynamicsProperties
        A multi-dynamics properties (which references a multi-dynamics class)
        which is to have linear sub-components merged where possible given
        matching dynamics and consistent parameters between sub-components
    validate : bool
        Whether to validate the merged dynamics or not (i.e. whether you trust
        the algorithm works), probably a good idea during prototyping phase
        but can slow the construction down significantly.
    """

    def __init__(self, multi_properties, validate=True):
        BaseVisitorWithContext.__init__(self)
        BaseDynamicsVisitor.__init__(self)
        self.multi_dynamics = multi_properties.component_class

        self.linear_sub_classes = []
        self.nonlinear_sub_classes = []

        self.state_var_map, self.param_map = self.get_name_maps(
            multi_properties)

        # Because we are iterating through child elements we can't remove and
        # add children as we go. So we save the objects to add and remove in
        # these lists along as the contexts they (are to) belong to
        self.to_remove = []
        self.to_add = []

        # Flatten multi dynamics so we can map and remove unrequired states +
        # dynamics
        self.merged_class = self.multi_dynamics.flatten()
        self.visit(self.merged_class)

        # Add and remove new and old children as required
        for obj, parent in self.to_remove:
            parent.remove(obj)
        for obj, parent in self.to_add:
            parent.add(obj)

        if validate:
            self.merged_class.validate()

        # Merge multi_properties
        self.merged = self._merge(multi_properties)

    def get_name_maps(self, multi_properties):
        # Sort sub-components into matching definitions and parameters used
        # in their time-derivatives
        candidates = defaultdict(list)
        for comp in multi_properties.sub_components:
            comp_class = comp.component_class
            # Cache linear and nonlinear sub-classes in list to save having to
            # check their linearity
            if comp_class in self.linear_sub_classes:
                is_linear = True
            elif comp_class in self.nonlinear_sub_classes:
                is_linear = False
            elif comp_class.is_linear():
                is_linear = True
                self.linear_sub_classes.append(comp_class)
            else:
                is_linear = False
                self.nonlinear_sub_classes.append(comp_class)
            if is_linear:
                candidates[comp.component_class].append(comp)

        # Used to hold mappings from state-variable and parameter names from
        # expanded original class to merged class
        state_var_map = {}
        param_map = {}

        # Group sub-components by component class
        for comp_class, comps in candidates.items():
            if len(comps) == 1:
                continue  # Skip as this is the only sub-comp of this class
            # Linear dynamics should only have 1 regime
            # FIXME: this is true unless just the output events change...
            assert comp_class.num_regimes == 1
            # Get set of parameters used in time-derivatives of the regime
            regime = next(comp_class.regimes)
            param_symbols = set(sympy.Symbol(s)
                                for s in comp_class.parameter_names)
            td_params = set(
                str(s) for s in chain(*(td.rhs_symbols
                                        for td in regime.time_derivatives))
                if s in param_symbols)

            # Group properties into groups with matching time derivative values
            # for parameters used in the state equations
            matching_td_props = defaultdict(list)
            for comp in comps:
                td_props = frozenset(comp.component[p] for p in td_params)
                matching_td_props[td_props].append(comp)

            for td_props, matching in matching_td_props.items():
                if len(matching) == 1:
                    continue  # Skip as no other sub-comp properties match
                if any(p.value.nineml_type != 'SingleValue' for p in td_props):
                    # Can't merge components with array or distributed values
                    continue
                ref = matching[0]
                for match in matching[1:]:
                    for param in td_params:
                        param_map[match.append_namespace(
                            param)] = ref.append_namespace(param)
                    for state_var in comp_class.state_variable_names:
                        state_var_map[match.append_namespace(
                            state_var)] = ref.append_namespace(state_var)
        return state_var_map, param_map

    def merge(self, multi_properties):
        state_var_map, param_map = self.get_name_maps(multi_properties)
        if state_var_map != self.state_var_map or param_map != self.param_map:
            raise NineMLCannotMergeException
        return self._merge(multi_properties)

    def _merge(self, multi_properties):
        flattened = multi_properties.flatten(component_class=self.merged_class,
                                             check_properties=False)
        # Removed merged properties and initial state values from merged
        # multiproperties
        for state_var in self.state_var_map:
            try:
                flattened.remove(flattened.initial_value(state_var))
            except NineMLNameError:
                pass  # Initial value not provided
        for param in self.param_map:
            flattened.remove(flattened.property(param))
        return flattened

    def default_action(self, obj, nineml_cls, **kwargs):  # @UnusedVariable
        pass

    def action_parameter(self, parameter, **kwargs):  # @UnusedVariable
        if parameter.name in self.param_map:
            self.to_remove.append((parameter, self.context.parent))

    def action_timederivative(self, time_derivative, **kwargs):  # @UnusedVariable @IgnorePep8
        if time_derivative.variable in self.state_var_map:
            self.to_remove.append((time_derivative, self.context.parent))

    def action_stateassignment(self, state_assignment, **kwargs):  # @UnusedVariable @IgnorePep8
        if state_assignment.variable in self.state_var_map:
            self.to_remove.append((state_assignment, self.context.parent))
            self.to_add.append(
                (StateAssignment(self.state_var_map[state_assignment.variable],
                                 state_assignment.rhs),
                 self.context.parent))

    def action_statevariable(self, state_variable, **kwargs):  # @UnusedVariable @IgnorePep8
        if state_variable.name in self.state_var_map:
            self.to_remove.append((state_variable, self.context.parent))
            self.to_add.append(
                (Alias(state_variable.name,
                       self.state_var_map[state_variable.name]),
                 self.context.parent))
