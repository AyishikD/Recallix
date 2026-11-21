class PredictiveRecall:
    def __init__(self, memory_manager):
        self.memory = memory_manager

    def preload_context(self, user_id, current_message, limit=5):
        """
        Predictive recall without an LLM: the topic is the message's schema
        (nearest prototype), and the predicted memories are the top lexical
        matches for the message among the user's stored memories.
        """
        vec = self.memory.semantic.embed(current_message)
        topic = self.memory.schema_classifier.classify(current_message, vec)
        predicted = self.memory.episodic.search(user_id, current_message, limit=limit)[:limit]
        return {"topic": topic, "predicted_memories": predicted}
