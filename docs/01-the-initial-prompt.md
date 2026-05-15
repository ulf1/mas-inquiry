Create a task plan and implementation plan for langgraph multi-agent system that processes a human inquiry. The worker agents tackle one question type, dimension, with primary focus and possible answer types. The goal of the multi-agent-system is to reduce uncertainty (increase certainty)

### Supervisor Agent
- A supervisor agent `InquirySupervisor` receive the inquiry as input (e.g. string), and returns the final output to the human. 
- Metric (supervisor): 
	- plus the Sqrt of the number of worker agents with at least 1 answer in their answer list.
	- plus the sum of the workers' metric.
- Goal (supervisor): Maximize the metric (Enrich information; Increase certainty)

### Worker Agents 
- Naming convention:`Inquiry{Dimension}` class (PascalCase), `inquiry_{Dimension}.py` (snake_case).
- Prompt: 
	- Identify up to {N=5} dissimilar, varied, and unique answers, that are most relevant to the inquiry. Keep each answer short and precise. 
	- For each answer, identify the answer type  (label as "unknown" if the given labels don't fit).
	- Compute the similarity score between the {N} answers. Similarity scores have values between 0.0 (not similar) and 1.0 (perfect match). If the two most similar answers have a) the same answer type, and b) a similarity score greater than {threshold=0.8}, then check the  two answers can be merged as "{combined answer}, e.g. {variant 1, variant 2}".
	- If an answer merger reduced the answer list from {N} to {N-1}, then try to find answer relevant to the inquiry that is dissimilar to the existing answer list - try this fill-up operation max. {R=2} times.
	- For each answer,  propose up to {M=3} related other inquiry dimensions (only if there is a related dimension).
- Pydantic outputs: 
	- store the {N} answers and their relevance scores for the inquiry as list, e.g. a=List[(answer, score)]
	- store the similarity scores as list of tuples, e.g. b=List[(i, j, score)] for i,j=0...N-1
	- store the connections to the {M} proposed related dimensions as list, e.g. c=List[(i, dimension_name)]
- Metric (workers): Compute a metric by adding the following terms
	- plus sqrt of list length a (number of answers)
	- plus sum of scores in a divided list length (mean relevance score)
	- minus sum of scores in b divided list length (mean similarity score)
	- plus list length c divided by list length a (average connections per answer)
- Goal (workers): Maximize the metric given limited N and M



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

### InquiryReplyMerger agent
- The InquiryReplyMerger agent takes 2 or more pydantic objects of a worker agent (see data structure of the worker agents is always an answer list, similarity score list, and connections list).
- When answer lists are concatenated, their index numbers must be adjusted in the similarity score list, and connections list accordingly.
- If the two most similar answers have a) the same answer type, and b) a similarity score greater than {threshold=0.8}, then check the  two answers can be merged as "{combined answer}, e.g. {variant 1, variant 2}". In case of a merge use the union of both connection lists.
- recompute the metric (worker)


### Graph Workflow
Design the langgraph workflow of the supervisor `InquirySupervisor`  as follows

1. During the first iteration, the supervisor distributes the human inquiry to all worker agents in parallel. 
2. The workers send their replies to the supervisor agent 
3. The supervisor agent computes its metric (supervisor).
5. Refinement loop:
    a. For each `dimension_name`  in the workers'  connections list, collect the linked answers, e.g. `Dict[dimension_name, List[(worker_answer, worker_dimension_name)]]`
	b. During the next iteration, the supervisor distributes the values of `dimension_name` to the corresponding `Inquiry{Dimension}` worker agent; convert the `List[(worker_answer, worker_dimension_name)]` value to a string beforehand.
	c. The workers send their replies to the supervisor agent.
	d. For each dimension merge the new reply (step c.) and previous reply (e.g. step 2. or c. of previous loop) and  into one pydantic object with the `InquiryReplyMerger`  agent.
	e. For each `dimension_name`  compare the merged result's metric (step d.) is higher than the previous reply's metric. If no, mark the previous result as final and deactivate the worker agent in the graph.
6. repeat step 5 till all worker agents have been deactivated or after {S=1} loops.
7. Based on the current worker replies, the `InquirySummary` worker agent generates a summary.
8. The supervisor agent returns the summary as string, and the `AgentState`  as results

The original human input should always be available to each worker agent.