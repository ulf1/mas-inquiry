---
name: multi_agent_system
description: Standards and patterns for building Supervisor-Worker multi-agent systems using LangGraph.
---

# Multi-Agent System (MAS) Architecture

This skill outlines the standard procedure for building **Supervisor-Worker** multi-agent systems using **LangGraph**. The architecture consists of specialized "Worker Agents", a routing "Supervisor Agent", and a "Graph" that orchestrates them.

**State Management:** Use `langgraph.checkpoint.memory.InMemorySaver` for in-memory checkpointing unless otherwise specified.

## 1. Interaction Guidelines

When the user asks to generate a MAS or a specific agent:
1.  **Identify the Role:** Is it a Worker, Supervisor, or the Graph itself?
2.  **Gather Context:**
    - **Workers:** What is the single specific purpose? What is the input/output schema?
    - **Supervisor:** Which workers does it manage?
3.  **LLM Integration:** Use `langchain-google-genai` and `ChatGoogleGenerativeAI` with `GEMINI_API_KEY`.

## 2. Recommended Directory Structure

Use a **Single-File Pattern** for agent definitions to keep related logic together. Keep the graph runtime logic separate and also use a single file per graph.

```text
src/
    agents/
        supervisors/     
            fitness.py   # Complete Supervisor (Schema + Prompt + Logic)
        workers/
            sport.py     # Complete Worker (Schema + Prompt + Logic)
            nutrition.py # Complete Worker (Schema + Prompt + Logic)
    graphs/
        fitness_bot.py   # LangGraph Implementation (Runtime) - State, Nodes, Workflow
```

---

## 3. Worker Agent Implementation

A Worker Agent is a specialized module responsible for a specific domain. It must expose its capabilities to a Supervisor and strictly structure its outputs.

### Code Structure (`src/agents/workers/{name}.py`)
Define the **Schema**, **Prompt**, and **Agent Class** in a single file.

```python
import jinja2
from pydantic import BaseModel, Field
from typing import Literal

# --- 1. SCHEMA ---
class SportLogSchema(BaseModel):
    activity: str = Field(description="Name of the activity (e.g., Running, Squats).")
    duration_min: int = Field(description="Duration in minutes.", ge=1)
    notes: str | None = Field(default=None, description="Optional notes.")

# --- 2. PROMPT ---
PROMPT_TEMPLATE = """
### ROLE
You are an expert in {{ domain }}.

### CONTEXT
Current Date: {{ current_date }}

### TASK
{{ task_description }}

### INPUT DATA
User Request:
\"\"\"
{{ user_message }}
\"\"\"

### FORMAT
Respond strictly with valid JSON conforming to the schema.
"""

# --- 3. AGENT CLASS ---
class SportAgent:
    name = "sport_log_extractor"
    description = "Parses unstructured workout logs into structured data. Use for any exercise-related inputs."
    output_schema = SportLogSchema

    @staticmethod
    def render_prompt(user_message: str, **kwargs) -> str:
        template = jinja2.Template(PROMPT_TEMPLATE)
        return template.render(
            domain="Sport & Fitness",
            user_message=user_message,
            **kwargs
        )
```

---

## 4. Supervisor Agent Implementation

The Supervisor's role is **Routing** and **Delegation**. It outputs a structured `RoutingDecision`.

### Code Structure (`src/agents/supervisors/{name}.py`)
Define the **Routing Schema**, **System Prompt**, and **Supervisor Class** in a single file.

```python
import jinja2
from pydantic import BaseModel, Field

# --- 1. ROUTING SCHEMA ---
class RoutingDecision(BaseModel):
    next_agent: str = Field(description="Name of the selected agent.")
    reasoning: str = Field(description="Reason for selection.")
    refined_instruction: str = Field(description="Cleaned instruction for the worker.")
    confidence_score: int = Field(description="1-10 confidence.")

# --- 2. SYSTEM PROMPT ---
# Crucial: Iterate over 'agents' to dynamically list capabilities
SUPERVISOR_PROMPT = """
### ROLE
You are the **Main Orchestrator**. Analyze user input and route to the correct agent.

### AVAILABLE AGENTS
{% for agent in agents %}
- **Name**: {{ agent.name }}
  **Description**: {{ agent.description }}
{% endfor %}

### ROUTING LOGIC
1. **Analyze**: Understand intent.
2. **Select**: Choose the best agent.
3. **Refine**: Clean up the input.

### INPUT DATA
User Message:
\"\"\"
{{ user_message }}
\"\"\"

### OUTPUT
Provide a valid JSON object matching the `RoutingDecision` schema.
"""

# --- 3. SUPERVISOR CLASS ---
class SupervisorAgent:
    @staticmethod
    def render_system_prompt(user_message: str, agents: list[object]) -> str:
        template = jinja2.Template(SUPERVISOR_PROMPT)
        return template.render(user_message=user_message, agents=agents)
```

---

## 5. LangGraph Assembly (The "Node-Bridge" Pattern)

Map the static "Agent Classes" into dynamic "LangGraph Nodes". Keep the shared state, node wrappers, and graph construction combined in a single file per graph.

### Graph Definition (`src/graphs/{name}.py`)

```python
import operator
import os
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver 
from dotenv import load_dotenv

# Import Single-File Agents
from src.agents.supervisors.fitness import SupervisorAgent, RoutingDecision
from src.agents.workers.sport import SportAgent

load_dotenv()

# --- 1. SHARED STATE ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str | None

# --- 2. NODE WRAPPERS ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY")
)

def create_agent_node(agent_class, custom_llm=None):
    runnable_llm = custom_llm or llm
    
    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        last_message = messages[-1]
        
        # 1. Render Prompt
        system_content = agent_class.render_prompt(
            user_message=last_message.content,
            current_date="2024-01-01" # Inject runtime context here
        )
        
        # 2. Bind Schema
        structured_llm = runnable_llm.with_structured_output(agent_class.output_schema)
        
        # 3. Invoke LLM
        response = structured_llm.invoke([
            SystemMessage(content=system_content),
            last_message
        ])
        
        # 4. Return result
        return {
            "messages": [AIMessage(content=str(response.model_dump()))],
            "next": "END"
        }
    return agent_node

# ... (standard supervisor_node implementation) ...
# e.g., def supervisor_node(state: AgentState) -> dict: ...

# --- 3. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("sport_log_extractor", create_agent_node(SportAgent))

workflow.set_entry_point("supervisor")

# Configure Routing
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "sport_log_extractor": "sport_log_extractor",
        "general_assistant": END 
    }
)
workflow.add_edge("sport_log_extractor", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

## 6. Best Practices

1.  **Single Responsibility:** Each agent file should contain everything needed to define that agent (Self-Contained).
2.  **Factory Pattern:** Use `create_agent_node` to avoid boilerplate.
3.  **Structured Routing:** Always use `RoutingDecision` for type-safe routing.
4.  **Loop Prevention:** Explicitly route workers to `END` unless multi-turn is required.
