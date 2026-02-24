---
name: agent_design_properties
description: Framework for defining agent properties, roles, and behaviors.
---

# Agent Design Properties

This skill defines the mandatory framework for specifying "Agent Properties" when designing new agents. Every agent must be defined with these 10 core properties.

## 1. Role & Identity
- **Concept:** Describe *who* the agent is (e.g., "Senior Python Engineer").
- **Expertise:** Specific domain knowledge.
- **Tone & Voice:** How the agent speaks.
- **Scope:** What the agent *can* do and what is *out of scope*.

## 2. Context & Environment
- **Background:** Project context, user proficiency.
- **Relationships:** Reporting lines (Supervisor) or delegation (Workers).
- **State Access:** Read/Write access to shared memory.

## 3. Goals & Objectives
- **Primary Goal:** Main outcome.
- **Secondary Goals:** Nice-to-haves.
- **Optimization:** What to maximize (speed, accuracy) or minimize (cost, complexity).

## 4. Constraints & Limitations
- **Negative Constraints:** What the agent must *never* do.
- **Resource Limits:** Time, token usage, file size.
- **Safety & Ethics:** Privacy and ethical rules.

## 5. Acceptance Criteria (Definition of Done)
- **Binary Checks:** Pass/Fail conditions.
- **Format Requirements:** JSON schemas, file types.
- **Completeness:** All requested parts present.

## 6. Tools & Capabilities
Explicitly list what the agent can use.
- **Tool Definitions:** Name, Purpose, Usage.
- **Access Level:** Read-only vs. Write.

## 7. Process & Reasoning
Instructions on *how* to think.
- **Chain of Thought:** Requirement to "think" before acting.
- **Step-by-Step:** Logical flow (Analyze -> Plan -> Execute).
- **Self-Correction:** Check results and retry.

## 8. Few-Shot Examples
Provide clear examples of inputs and expected outputs.
- **Input:** Sample user query.
- **Thought:** Internal reasoning.
- **Output:** Expected response format.

## 9. Termination Conditions
Rules for stopping.
- **Success:** Criteria met.
- **Failure:** Max retries exceeded.
- **Hard Limits:** Max steps.

## 10. Output Format
Strict definition of response structure.

### Type A: Human-Facing
For chat agents.
```markdown
## Analysis
[Thinking]
## Action
[Tools]
## Result
[Deliverables]
```

### Type B: Worker Node
For system agents (JSON).
```json
{
  "key": "value"
}
```
