from langgraph.graph import StateGraph
from langgraph.graph.message import MessageGraph
from utils.models import UserState
from nodes import embed_store, intent_parser, log_workout, suggest_plan, motivation, progress, update_goal, \
    recovery_check, fallback, response_builder

workflow = StateGraph(UserState)

workflow.add_node("EmbedAndStore", embed_store.embed_and_store_node)
workflow.add_node("ParseIntent", intent_parser.parse_intent_node)
workflow.add_node("LogWorkout", log_workout.log_workout_node)
workflow.add_node("PlanSuggestion", suggest_plan.suggest_plan_node)
workflow.add_node("Motivation", motivation.motivation_node)
workflow.add_node("Progress", progress.progress_node)
workflow.add_node("UpdateGoal", update_goal.update_goal_node)
workflow.add_node("RecoveryCheck", recovery_check.recovery_check_node)
workflow.add_node("Fallback", fallback.fallback_node)
workflow.add_node("BuildResponse", response_builder.response_builder_node)

workflow.set_entry_point("EmbedAndStore")
workflow.add_edge("EmbedAndStore", "ParseIntent")

workflow.add_conditional_edges(
    "ParseIntent",
    lambda state: state.intent,
    {
        "log_workout": "LogWorkout",
        "ask_for_plan": "PlanSuggestion",
        "motivation": "Motivation",
        "ask_progress": "Progress",
        "goal_update": "UpdateGoal",
        "rest_check": "RecoveryCheck",
        "unknown": "Fallback"
    }
)


for node in ["LogWorkout", "PlanSuggestion", "Motivation", "Progress", "UpdateGoal", "RecoveryCheck", "Fallback"]:
    workflow.add_edge(node, "BuildResponse")

workflow.set_finish_point("BuildResponse")
app = workflow.compile()

