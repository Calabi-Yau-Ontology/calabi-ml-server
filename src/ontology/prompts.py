PROPOSE_SYSTEM_PROMPT = """You are an ontology engineer.
Return ONLY valid JSON that matches the ProposeResponse schema.
Never include markdown, comments, or extra text.
"""

PROPOSE_USER_PROMPT_TEMPLATE = """
[Context]
We build a personal knowledge graph from calendar titles (mentions only).
Do NOT assume attendees, GPS, or true duration. No Wikidata expansion.

[Task]
Given CQs and the current taxonomy snapshot, propose minimal additions:
- oClassesToAdd (leaf-first; avoid new roots)
- subclassEdgesToAdd
- classifications (conceptKey -> oClassId with confidence)
- queryTemplates (read-only querySketch + params + outputSchema)

[Hard Rules]
- If unsure, keep classification at higher parent level.
- Do NOT invent new data fields.
- Do NOT output analysis metrics as OProp (distribution/trend/score).
- For existing_only mode (in /classify), do NOT add new classes or edges.

[Payload JSON]
{payload}
"""

CLASSIFY_SYSTEM_PROMPT = """You are a taxonomy classifier.
Return ONLY valid JSON that matches the ClassifyResponse schema.
Never include markdown, comments, or extra text.
"""

CLASSIFY_USER_PROMPT_TEMPLATE = """
[Context]
We classify concept mentions from calendar titles into the existing OClass tree.
No Wikidata. If ambiguous, choose the higher-level class.

[Task]
Classify each concept into an OClass with confidence.
- If mode == "existing_only": no new classes or edges allowed.
- If mode == "allow_new_leaf": new leaf classes and subclass edges are allowed.
- Use context fields when available:
  - sourceText (original title)
  - normalizedTextEn
  - surface + span
  - examples[] (recent titles)

[Payload JSON]
{payload}
"""
