# Multi-Agent System (MAS) with LangGraph


## Setup

To set up the development environment, run the following commands:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

## Add Gemini API Key

```bash
export GEMINI_API_KEY="your-api-key"
```

or write it in an `.env` file.


## Unit Tests

```bash
PYTHONPATH=. uv run pytest tests/
```


## CLI

```bash
python -m src.cli --query "Correct push-ups?"
python -m src.cli --query "Healthier options than burger and fries"
python -m src.cli --query "Research the current state of AI agents"
```


## Preparation
This system was mainly designed with the help of AI tools, and funnily enough, it is about designing a multi-agent system to do AI stuff.

#### Step 1: Product research
Ask chat-based AI tools the following questions:
- "what kind of interactions or dialogues are known between human, ai, data?"
- "Assume a string of text as input, what are possible ways to process it to an output? The system is then looking for what, why, ..."
- ... and so forth ...

The results are the three tables in the section "Question Workers".

#### Step 2: System design
Some system design decisions, e.g. LangGraph with Gemini LLM API calls, input via CLI.
Setup `pyproject.toml` and run `uv`.


#### Step 3: IDE+AI tool selection
Select an IDE+AI tool. I selected [Google Antigravity](https://antigravity.dev/).
In the next step, the copilot instructions markdowns were transformed into Antigravity Skills (See `.agent/skills/...`).
I already had copilot instructions for specific languages and frameworks (e.g. Python, LangGraph, FastAPI), 
and tasks (e.g. Code Review, AI agent design properties, specific multi-agent patterns).

You literally prompt the Antigravity AI assistant:
- "Refactor the copilot instructions into Antigravity Skills"
- "Check if files have redundant information and remove it"
- "Propose missing best practices"
- "Ask for clarification if something is unclear or ambiguous"
- ... and so forth ...

I think the [MECE principle](https://en.wikipedia.org/wiki/MECE_principle) is a good guiding principle for this process:
The points should cover ideally 100% (collectively exhaustive) but the points should have minimal overlap (mutually exclusive).

Given my idea about the system architecture, it took me two evenings to refine just the instructions files. No pressure here.
Otherwise the Coding Agent would just flip coins, and branch into wanted or unwanted directions. 

#### Step 4: AI Coding on Autopilot
The [the-initial-prompt.md](the-initial-prompt.md) is what was copy-pasted into the Antigravity AI assistant.
With hindsight, the initial prompt looks messy and chaotic; I wrote it but find it hard to understand now...
It took me around 3 hours to write this (immature) initial prompt (draft).
Finally, it took Antigravity a few minutes to generate a working prototype on autopilot.
And the thing actually worked as intended. 

Honestly, I could have just stopped there. Skim over it, and ship it.

#### Step 5: Manual Coding
I was curious how long it would take to refactor the system without an AI assistant just to understand what the AI assistant coded.
It took me around 3 evenings. 
My refactored version is posted in this repo. 
It's basically double-checking the AI-generated solution but I didn't really find any major issues that would have prevented the system from working (There was just one bug).

#### Conclusion:
- **Product research** can happen with Chat-AI-tools, e.g. Deep Research modes.
- **Specification** doesn't happen in Kanban boards anymore - it happens in Markdown files (Is Kanban dead? Is Scrum a relic of the past? Why writing git issues if we can fix it immediately?)
- Spend time on refining general **Instructions** for the AI assistant, e.g. prompt to remove ambiguities and contradictions.
- Spend time on the large **initial prompt** - if it's about math on a computer, then prompt at least pseudocode and algorithmic descriptions as you would actually want to code it.
- If the initial code generation prompt fails, then don't fix the code, **fix the instructions and initial prompt, and start over again**. We humans don't code anything anymore in the AI assistant world! (see step 4)
- If something really needs security checks, is mission critical, or otherwise, then good programming skills might be necessary for **Debugging**, **Reverse Engineering**, or **Security Analysis** (see step 5). However, is all software so important? 

Before coding agents, software development was so expensive and way undersupplied, that humanity probably doesn't know the actual demand for software.



## Inquiry Framework

### Graph
It's basically like sparse connectivity neural networks. 
- input layer: The Questions Workers specialized on a topic all get a string as input, and produce multiple answers with each linked to a topic. 
- hidden layer: Each answer is the passed to linked Question Worker, and so on. 
- output layer: Summarize all answers from all Question Workers into a output string.

Sure there further details to the graph:
- Depth: up to 3 hidden layers are partially stacked.
- Skip connections: Each Question Worker's previous state and output state are merged.
- Dynamic sparsity net: The number of answers is limited, and thus it forces sparse connectivity but not in a static way.
- Variable depth/recursion: If a skip connection doesn't improve a metric, the particular Question Worker is deactivated.

While writing the initial prompt, I wondered if all the biology analogies and neural network design ideas from the past were never supposed to just use floating point numbers as inputs, weights, and outputs (I bet Schmidhuber wrote a paper about all of that...) 


### Question Workers
I think linguists call this stuff "concepts" (but maybe I'm wrong). Each Question Worker focuses on searching for the what, who, when, where, why, and how of a given string. In grammar and morphology (linguistics) we have PoS (Part of Speech) tags, and NER (Named Entity Recognition) tags, and so forth for that. However, here it's more about jeopardy. A human might prompt the input string "Correct push-ups?" which provides almost nothing. For example, the Question Worker "Causal" (Why?) searches for "Why?" information related to the input string. 


The **base** questions types:

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


Additional **multimodal** questions types:

| **Question Type**   | **Dimension**     | **Primary Focus**  | **Answer Types (Brackets)**                 | **Contextual Utility (Short Description)**                                         |
| ------------------- | ----------------- | ------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------- |
| **How It Sounds?**  | **Auditory**      | Sonic Signature    | (Pitch, timbre, volume, resonance)          | **The Signal:** Analyzes acoustic data for stress, health, or identification.      |
| **How It Looks?**   | **Visual**        | Appearance         | (Color, shape, texture, composition)        | **The Aesthetic:** Decodes visual hierarchies and surface-level truths.            |
| **How It Moves?**   | **Kinetic**       | Dynamic State      | (Velocity, trajectory, fluidity, blur)      | **The Momentum:** Tracks change-over-time within a single frame or event.          |
| **How It Feels?**   | **Tactile**       | Physical Surface   | (Pressure, temperature, friction, weight)   | **The Haptic:** Evaluates the physical reality and material "touch" of data.       |
| **What Signal?**    | **Measured**      | Data Telemetry     | (Voltage, frequency, latency, variance)     | **The Precision:** Uses objective instrumentation to bypass human bias.            |
| **What Mood?**      | **Affective**     | Emotional State    | (Valence, arousal, sentiment, vibe)         | **The Atmosphere:** Gauges the emotional resonance or psychological tone.          |


The **hidden** dimension questions:

| **Question Type**   | **Dimension**     | **Primary Focus** | **Answer Types (Brackets)**              | **Contextual Utility (Short Description)**                                         |
| ------------------- | ----------------- | ----------------- | ---------------------------------------- | ---------------------------------------------------------------------------------- |
| **What's Missing?** | **The Void**      | Omission          | (Absence, silence, blind spot, gap)      | **The Negative Space:** Essential for forensics; looks for what _should_ be there. |
| **Whose View?**     | **Perspective**   | Bias & Lens       | (Framing, subjectivity, cultural filter) | **The Lens:** Corrects for the observer effect and subjective distortion.          |
| **What If?**        | **Hypothetical**  | Counterfactual    | (Simulation, alternative, prediction)    | **The Divergence:** Explores potential realities and stress-tests outcomes.        |
| **Is it Right?**    | **Ethical**       | Moral Value       | (Equity, privacy, consent, integrity)    | **The Conscience:** Filters the inquiry through legal and moral constraints.       |
| **How Likely?**     | **Probabilistic** | Certainty         | (Confidence level, risk, entropy)        | **The Odds:** Quantifies the reliability of the entire data set.                   |
| **What's Under?**   | **Subtext**       | Latent Meaning    | (Innuendo, symbology, unspoken rule)     | **The Undercurrent:** Decodes the symbolic or hidden "meta-narrative."             |


### The InquiryOther Worker
When a Question Worker is tasked to generate answers, the worker is supposed to associate each answer with one of the given question types (or topics, or concepts), so that the answer is passed to the next Question Worker in the graph. During the test runs, the Question Workers sometimes produced new question types. These made sense on visual inspection, but I didn't add them to the base question types. Instead, the `InquiryOther` Question Worker was added, which is like a residual term in a linear regression model, a catch-all node - it makes an input layer or hidden layer MECE (mutually exclusive, collectively exhaustive).


### Merge Worker
The Prompt in `inquiry_base.py` and `inquiry_reply_merger.py` are very similar.
The `MERGER_PROMPT` is more about reducing the number of answers to given maximum number of answers (and usually the number of connections to other Question Workers).
The `BASE_PROMPT_TEMPLATE` is more about near duplicate detection and merging of answers.
The prompted algorithm is the same: It's a classic optimization goal to maximize the sum of relevance scores while minimizing the similarity between answers.

=> Any agent without **goal** is not really an agent. (It's just a function)


### Summary Worker
Takes answers and generates a summary for human consumption.




## Changes

### 2026-02-24
- forked from may mas-ghost3 project
- initial prompt copied from Obsidian
- README.md updated with comments
