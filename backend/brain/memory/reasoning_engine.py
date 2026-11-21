import re


class ReasoningEngine:
    """
    Multi-hop expansion for recall.

    Alias resolution and synonym handling used to be hardcoded tables here;
    aliases now come from stored "X is also known as Y" facts
    (RecallEngine._resolve_aliases) and synonyms are handled by embedding
    similarity, so only structural hop detection remains.
    """

    _HOP = re.compile(
        r"\b(?:contains|includes|is\s+powered\s+by|powered\s+by|depends\s+on)\s+"
        r"(?:the\s+)?([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})"
    )

    def __init__(self, recall_engine=None):
        self.recall_engine = recall_engine

    def identify_multi_hop_opportunities(self, memories, analysis):
        """
        If a top memory says the asked-about entity contains / is powered by /
        depends on another named entity, return that entity as a second-hop
        target — unless the question already names it.
        """
        asked = set(analysis.get("strong_entities", []))
        hops = []
        for m in memories:
            for target in self._HOP.findall(m.get("content", "")):
                target_id = re.sub(r"[^a-z0-9]+", "", target.lower())
                if target_id and target_id not in asked:
                    hops.append(target)
        return list(dict.fromkeys(hops))
