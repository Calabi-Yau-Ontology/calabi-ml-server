PROPOSE_SYSTEM_PROMPT = """You are an ontology engineer.
Return ONLY valid JSON that matches the ProposeResponse schema.
Never include markdown, comments, or extra text.
"""

PROPOSE_USER_PROMPT_TEMPLATE = """
[Context]
We build a personal knowledge graph from calendar titles (mentions only).
Do NOT assume attendees, GPS, or true duration. No Wikidata expansion.
Data graph (available for queryTemplates):
- (:User)-[:OWNS_EVENT]->(:Event)
- (:Event)-[:MENTIONS]->(:Concept)
- (:User)-[:RELATED_TO]->(:Concept)
- (:Event)-[:HAS_ACTIVITY]->(:OClass)  // event-level activity classification

[Task]
Given CQs and the current taxonomy snapshot, propose minimal additions:
- oClassesToAdd (leaf-first; avoid new roots)
- subclassEdgesToAdd
- classifications (conceptKey -> oClassId with confidence)
- queryTemplates (read-only querySketch + params + outputSchema)

[Hard Rules]
- If unsure, OMIT the classification (leave unclassified). Do NOT choose a parent.
- CQs are NOT concepts. Do NOT classify CQs or turn CQ text into OClasses.
- Do NOT invent new data fields.
- Do NOT output analysis metrics as OProp (distribution/trend/score).
- For existing_only mode (in /classify), do NOT add new classes or edges.
- You MUST output at least one queryTemplate per CQ.
  - If a CQ is truly impossible, add a warning explaining why and omit only that CQ's template.

[Payload JSON]
{payload}
"""

CLASSIFY_SYSTEM_PROMPT = """You are a taxonomy classifier.
Return ONLY valid JSON that matches the ClassifyResponse schema.
Never include markdown, comments, or extra text.
"""

CLASSIFY_ROOT_SYSTEM_PROMPT = """You are a taxonomy root selector.
Return ONLY valid JSON that matches the RootSelectionResponse schema.
Never include markdown, comments, or extra text.
"""

CLASSIFY_ROOT_USER_PROMPT_TEMPLATE = """
[Context]
Select plausible ROOT OClasses for each concept and for the event title.
You will be given ONLY root classes (top-level taxonomy nodes).
If eventRootOClasses is provided, use ONLY those for the event title.
If conceptRootOClasses is provided, use ONLY those for concept roots.

[Task]
- For each concept: choose up to 5 roots with confidence (0..1).
- For the event title: choose up to 5 roots with confidence.
- Prefer recall (include multiple plausible roots if unsure).
- If nothing fits, return an empty list.

[Payload JSON]
{payload}
"""

CLASSIFY_USER_PROMPT_TEMPLATE = """
[Context]
We classify concept mentions from calendar titles into the existing OClass tree.
No Wikidata. If ambiguous, OMIT the classification.

[Task]
Classify each concept into a LEAF OClass selected from the provided candidate subtrees.
Also classify the event title into a LEAF Activity OClass if event candidates are provided.
- Choose ONLY leaf ids from the provided candidate lists.
- If you cannot confidently choose a leaf, OMIT the item (do not output a classification for it).
- Never invent Other/Unknown or any forced placeholder leaf.
- If mode == "existing_only": do NOT add new classes or edges.
- If mode == "allow_new_leaf": new leaf classes and subclass edges are allowed.
- Respect scope:
  - scope == "concept": do NOT return eventActivities.
  - scope == "event": you may ignore concept classifications.
  - scope == "both": return both when candidates exist.
- Use context fields when available:
  - sourceText (original title)
  - normalizedTextEn
  - surface + span
  - examples[] (recent titles)
- If scope includes event, always return `eventActivities` (empty array if no activity).

[Payload JSON]
{payload}
"""
