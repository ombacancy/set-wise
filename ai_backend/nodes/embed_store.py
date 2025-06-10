from utils.vdb import store_text


def embed_and_store_node(state):
    store_text(state.user_input, {"user_id": state.user_id})
    return state
