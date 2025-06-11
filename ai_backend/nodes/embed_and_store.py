from schema import IntentPayload
from vdb import store_log
from datetime import datetime


class EmbedAndStoreNode:
    def __call__(self, state: IntentPayload) -> IntentPayload:
        metadata = {
            "user_id": state.user_id,
            "intent": "unknown",  # set post-intent parsing
            "timestamp": datetime.utcnow().isoformat()
        }
        store_log(state.user_id, state.raw_input, metadata)
        return state  # pass unchanged to next node
