# langgraph_flow.py

from langgraph.graph import StateGraph, END
from schema import IntentPayload
from nodes.parse_intent import ParseIntentNode
from nodes.log_workout import LogWorkoutNode
from nodes.plan_suggestion import PlanSuggestionNode
from nodes.motivation import MotivationNode
from nodes.progress import ProgressNode
from nodes.goal_update import GoalUpdateNode
from nodes.recovery_check import RecoveryCheckNode
from nodes.fallback import FallbackNode

# Step Nodes
parse_intent_node = ParseIntentNode()
log_workout_node = LogWorkoutNode()
plan_node = PlanSuggestionNode()
motivation_node = MotivationNode()
progress_node = ProgressNode()
goal_update_node = GoalUpdateNode()
recovery_node = RecoveryCheckNode()
fallback_node = FallbackNode()


# Router
def route_intent(state: IntentPayload):
    intent = state.intent
    return {
        "log_workout": "log_workout",
        "get_plan": "plan_suggestion",
        "motivate": "motivation",
        "check_progress": "progress",
        "update_goal": "goal_update",
        "recovery_check": "recovery_check",
    }.get(intent, "fallback")


# Define the LangGraph
def build_graph():
    builder = StateGraph(IntentPayload)

    builder.add_node("parse_intent", parse_intent_node)
    builder.add_node("log_workout", log_workout_node)
    builder.add_node("plan_suggestion", plan_node)
    builder.add_node("motivation", motivation_node)
    builder.add_node("progress", progress_node)
    builder.add_node("goal_update", goal_update_node)
    builder.add_node("recovery_check", recovery_node)
    builder.add_node("fallback", fallback_node)

    builder.set_entry_point("parse_intent")

    builder.add_conditional_edges(
        "parse_intent",
        route_intent,
        {
            "log_workout": "log_workout",
            "get_plan": "plan_suggestion",
            "motivate": "motivation",
            "check_progress": "progress",
            "update_goal": "goal_update",
            "recovery_check": "recovery_check",
            "fallback": "fallback",
        }
    )

    # All leaf nodes go to END
    for node in ["log_workout", "plan_suggestion", "motivation", "progress", "goal_update", "recovery_check",
                 "fallback"]:
        builder.add_edge(node, END)

    return builder.compile()
