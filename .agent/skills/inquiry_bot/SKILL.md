---
description: Instructions and architectural analysis of the Universal Inquiry Bot.
---

# Universal Inquiry Bot

The Universal Inquiry Bot is designed to dissect a human inquiry across multiple analytical dimensions (e.g., Causal, Temporal, Structural) using a sparsely-connected multi-agent architecture inspired by biological neural networks.

## 1. LangGraph Architecture

The system is orchestrated using `langgraph` as a state machine (`StateGraph`).

### Shared State (`AgentState`)
- **`inquiry`**: The original user string or question.
- **`active_workers`**: List of currently active dimension workers.
- **`deactivated_workers`**: Workers that failed to yield improved metrics and were deactivated.
- **`loop_count`**: Counter controlling the recursion/skip connections depth (up to 2 iterations).
- **`stop`**: Boolean flag indicating if the iteration should stop.
- **`worker_replies`**: A composite dictionary tracking answers and connections mapped to each dimension.
- **`summary`**: The final synthesized output.

### Graph Nodes & Workflow
1. **Input Layer (`prelim_nodes`)**: The system broadcasts the user `inquiry` to all `active_workers` concurrently via `ThreadPoolExecutor`. Each worker agent produces an initial set of answers related to its dimension's focus and suggests links (skip connections) to other dimensions.
2. **Hidden Layers (`cross_nodes`)**: Using the connections defined in the input layer, the overarching graph feeds answers from source dimensions as `additional_context` into the LLM prompts of target dimensions. 
    - The target workers generate a new set of focused/refined answers.
    - An `InquiryReplyMerger` is then invoked to merge the newly generated context-aware answers with the previously recorded answers for that exact dimension.
    - A **worker metric** is calculated (evaluating relevance score versus similarity score). If the context-merged result does not improve the metric over the previous result, the worker is added to `deactivated_workers` and practically frozen.
    - This conditionally loops if `loop_count < 2`.
3. **Output Layer (`summarizer`)**: All consolidated `worker_replies` are given to a single `InquirySummary` node, which relies on a separate LLM prompt to digest the answers and summarize the final output.

The main orchestration logic is defined in [inquiry_bot.py](file:///home/ulf/repos/antigravity/mas-ghost3/src/graphs/inquiry_bot.py).

## 2. Worker Mechanics

Each worker agent inherits from `BaseInquiryWorker` and represents a specific grammatical or analytical concept (e.g., `InquiryCausal` for "Why?", `InquiryTemporal` for "When?"). There are over 20 pre-configured base dimensions.

**Workflow within a Worker**:
- **Prompt Generation**: The worker uses `BASE_PROMPT_TEMPLATE` via Jinja2, injected with its specific `dimension`, `primary_focus`, and valid `answer_types` (e.g., *color*, *shape*, *fluidity*, *cause*).
- **Retrieval & Structuring**: The LLM acts to populate an `answers_list` where each structural answer yields:
  - Textual content
  - Answer Type
  - Relevance Score (0.0 to 1.0)
- **Connections**: The worker specifies skip connections to other dimension graphs using a `connections_list`.
- **Evaluation Metric (`calculate_worker_metric`)**: Reponses are evaluated via an algorithmic metric calculation:
  $Metric = \sqrt{\text{num\_answers}} + \text{average\_relevance} - \text{average\_similarity} + \text{average\_connections\_per\_answer}$
  This purposefully incentivizes workers to hunt for a high volume of relevant, highly distinct answers that richly interlink with other complementary dimensions. 

There is also an **`InquiryOther`** worker. When an LLM outputs unstructured answers with unmapped or hallucinated dimensions, this catch-all node is allocated so no context escapes processing boundaries.

## 3. Algorithmic Processing via LLM Prompt (Pseudocode)

The LLM is explicitly instructed to execute rigid optimization algorithms directly in its prompt. These logic loops handle deep deduplication and merging natively before ultimately outputting the specified JSON schema for `WorkerReply`.

### A. Worker Generation (`inquiry_base.py`)

For isolated worker exploration:
```text
loops = 0
answers_list = []

While loops < max_fillups and len(answers_list) < max_answers:
    1) Generate distinct answers most relevant to the target dimension focus. For each:
       - Compute relevance score against the inquiry.
       - Identify the answer type.
       - Add to answers_list.
       
    2) Compute similarity scores between answers of the identical answer type.
    
    3) Find the pair of answers (X, Y) with the highest similarity score.
       If similarity(X, Y) > threshold:
           - Merge X and Y into a new structured answer Z.
           - Compute a new relevance score for Z.
           - If relevance(Z) > relevance(X) AND relevance(Z) > relevance(Y):
               Remove X and Y from answers_list.
               Add Z to answers_list.
               Discard similarity score between X and Y.
               
    loops += 1

6) For each surviving answer, propose up to `max_connections` matching to other valid framework Dimensions.
Return (answers_list, similarity_scores, connections_list)
```

### B. Reply Merger (`inquiry_reply_merger.py`)

When aggregating a merged reply across recursive graph layers, the system depends on the `InquiryReplyMerger` prompt instruction set to reliably squash the data format into defined length constraints:

```text
loops = 0
answers_list = concat(data1.answers_list, data2.answers_list)
connections_list = concat(data1.connections_list, data2.connections_list)

While loops < max_removals and len(answers_list) > max_answers:

    2) Compute similarity scores between existing answers of identical answer types.
    
    3) Find the pair of answers (X, Y) with the highest similarity score.
       If similarity(X, Y) > threshold:
           - Merge X and Y into a new, encompassing answer Z.
           - Compute relevance score for Z.
           - If relevance(Z) > relevance(X) AND relevance(Z) > relevance(Y):
               Remove X and Y from answers_list.
               Add Z to answers_list.
               Remove similarity score between X and Y.
               Drop respective connections related to X and Y.
               Add new contextual connection for Z.
               
    loops += 1

Return (answers_list, similarity_scores, connections_list)
```

## Note for Antigravity AI Assistants

When extending or inspecting the `inquiry_bot` module, adhere to the architectural expectations: 
- Understand that the state transitions of the inner nodes (Input to Hidden Layer to Hidden Layer) actively evaluate the custom geometric heuristic defined in `calculate_worker_metric` before persisting node output.
- Structural algorithms are natively embedded in the prompt context of `BASE_PROMPT_TEMPLATE` and `MERGER_PROMPT`. Modifying deduplication constraints or similarity thresholds directly modifies cognitive depth and network connectivity sparsity.
- A global metric for the entire inquiry set is calculated by `InquirySupervisor.calculate_metric` in [inquiry_supervisor.py](file:///home/ulf/repos/antigravity/mas-ghost3/src/agents/supervisors/inquiry_supervisor.py).


## Workers


| **Question Type**   | **Dimension**     | **Primary Focus**  | **Answer Types (Brackets)**                 | **Contextual Utility (Short Description)**                                         |
| ------------------- | ----------------- | ------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------- |
| **What?**           | **Content**       | Essence & Identity | (Object, entity, definition, category)      | **The Anchor:** Establishes the baseline facts and subject matter.                 |
| **Who?**            | **Agent**         | Actors & Roles     | (Person, animal, stakeholder, victim)       | **The Identity:** Assigns agency, responsibility, and human impact.                |
| **When?**           | **Temporal**      | Timing & Flow      | (Time, duration, frequency, era, pace)      | **The Chronology:** Maps the event onto a timeline to find patterns.               |
| **Where?**          | **Spatial**       | Location & Setting | (Context, environment, coordinate, layout)  | **The Perimeter:** Defines the physical or digital boundaries of the event.        |
| **Why?**            | **Causal**        | Motivation & Logic | (Intent, cause, rationale, justification)   | **The Catalyst:** Uncovers the underlying "engine" driving the behavior.           |
| **How?**            | **Procedural**    | Method & Flow      | (Mechanism, process, instrument, manner)    | **The Mechanics:** Details the specific steps taken to achieve the result.         |
| **How Much?**       | **Quantitative**  | Magnitude          | (Scale, mass, budget, threshold, count)     | **The Scale:** Measures the volume or intensity of the subject.                    |
| **Which?**          | **Selective**     | Distinction        | (Selection, preference, criteria, priority) | **The Choice:** Identifies why one path was taken over another.                    |
| **So What?**        | **Impact**        | Consequence        | (Implication, significance, value, risk)    | **The Gravity:** Determines the meaningfulness or severity of the data.            |
| **What Next?**      | **Actionable**    | Iteration          | (Action, recommendation, follow-up)         | **The Vector:** Points toward the necessary response or next phase.                |



| **Question Type**   | **Dimension**     | **Primary Focus**  | **Answer Types (Brackets)**                 | **Contextual Utility (Short Description)**                                         |
| ------------------- | ----------------- | ------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------- |
| **How It Sounds?**  | **Auditory**      | Sonic Signature    | (Pitch, timbre, volume, resonance)          | **The Signal:** Analyzes acoustic data for stress, health, or identification.      |
| **How It Looks?**   | **Visual**        | Appearance         | (Color, shape, texture, composition)        | **The Aesthetic:** Decodes visual hierarchies and surface-level truths.            |
| **How It Moves?**   | **Kinetic**       | Dynamic State      | (Velocity, trajectory, fluidity, blur)      | **The Momentum:** Tracks change-over-time within a single frame or event.          |
| **How It Feels?**   | **Tactile**       | Physical Surface   | (Pressure, temperature, friction, weight)   | **The Haptic:** Evaluates the physical reality and material "touch" of data.       |
| **What Signal?**    | **Measured**      | Data Telemetry     | (Voltage, frequency, latency, variance)     | **The Precision:** Uses objective instrumentation to bypass human bias.            |
| **What Mood?**      | **Affective**     | Emotional State    | (Valence, arousal, sentiment, vibe)         | **The Atmosphere:** Gauges the emotional resonance or psychological tone.          |



| **Question Type**   | **Dimension**     | **Primary Focus** | **Answer Types (Brackets)**              | **Contextual Utility (Short Description)**                                         |
| ------------------- | ----------------- | ----------------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
| **What's Missing?** | **The Void**      | Omission          | (Absence, silence, blind spot, gap)      | **The Negative Space:** Essential for forensics; looks for what _should_ be there. |
| **Whose View?**     | **Perspective**   | Bias & Lens       | (Framing, subjectivity, cultural filter) | **The Lens:** Corrects for the observer effect and subjective distortion.          |
| **What If?**        | **Hypothetical**  | Counterfactual    | (Simulation, alternative, prediction)    | **The Divergence:** Explores potential realities and stress-tests outcomes.        |
| **Is it Right?**    | **Ethical**       | Moral Value       | (Equity, privacy, consent, integrity)    | **The Conscience:** Filters the inquiry through legal and moral constraints.       |
| **How Likely?**     | **Probabilistic** | Certainty         | (Confidence level, risk, entropy)        | **The Odds:** Quantifies the reliability of the entire data set.                   |
| **What's Under?**   | **Subtext**       | Latent Meaning    | (Innuendo, symbology, unspoken rule)     | **The Undercurrent:** Decodes the symbolic or hidden "meta-narrative."             |
