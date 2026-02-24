import operator
import importlib
import os
from typing import TypedDict, Annotated
from concurrent.futures import ThreadPoolExecutor
from src.agents.workers.inquiry_base import WorkerReply

import time
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# llm client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

import json

from src.agents.workers.inquiry_reply_merger import InquiryReplyMerger
from src.agents.workers.inquiry_base import calculate_worker_metric, ALL_DIMENSIONS, InquiryOther
from src.agents.workers.inquiry_summary import InquirySummary


# import util for worker class
def get_worker_class(dimension: str):
    dim_snake = dimension.lower().replace(" ", "_").replace("'", "")
    dim_class = dimension.replace(" ", "").replace("'", "")
    try:
        module = importlib.import_module(f"src.agents.workers.inquiry_base")
        return getattr(module, f"Inquiry{dim_class}")
    except (ImportError, AttributeError):
        obj = InquiryOther()
        obj.set_defintion(dimension)
        return obj


# --- 1. SHARED STATE ---
def merge_dict(val1: dict, val2: dict) -> dict:
    if not val1:
        val1 = {}
    if not val2:
        val2 = {}
    res = dict(val1)
    res.update(val2)
    return res


class AgentState(TypedDict):
    inquiry: str
    active_workers: list[str]
    deactivated_workers: list[str]
    loop_count: int
    stop: bool
    worker_replies: Annotated[dict[str, WorkerReply], merge_dict]
    summary: str | None




# --- 2. NODE WRAPPERS ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY"),
)



def init_node(state: AgentState):
    return {
        "active_workers": ALL_DIMENSIONS,
        "deactivated_workers": [],
        "loop_count": 0,
        "stop": False,
        "summary": None
    }


HUMAN_PROMPT = "Please process the inquiry and provide the structured list as requested."

def prelim_nodes(state: AgentState):
    # loop over state.active_workers (ALL_DIMENSIONS) to create batch of inline requests
    results = {}
    
    def process_worker(name):
        worker_class = get_worker_class(name)  # get Inquiry<Dimension> class
        system_content = worker_class.render_prompt(state["inquiry"]) # create prompt
        structured_llm = llm.with_structured_output(worker_class.output_schema) # set output schema
        response = structured_llm.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=HUMAN_PROMPT)
        ]) # invoke llm
        return name, response

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_worker, name): name for name in state["active_workers"]}
        for future in futures:
            name = futures[future]
            _, response = future.result()
            results[name] = response
            
    return {"worker_replies": results}


MERGER_PROMPT = "Please process the previous texts and merge the overlapping contents efficiently."

def cross_nodes(state: AgentState):
    # use the answers
    new_inputs = {}
    for from_dim, reply in state["worker_replies"].items():
        for conn in reply.connections_list:
            # skip deactivated agents
            to_dim = conn.dimension_name
            if to_dim in state["deactivated_workers"]:
                continue
            # from: get the answer from the reply
            ans = reply.answers_list[conn.i].model_dump()
            ans["from_dim"] = from_dim
            # to: save
            if to_dim not in new_inputs:
                new_inputs[to_dim] = [ans]
            else:
                new_inputs[to_dim].append(ans)
    
    # combine previous answer to prompt
    results = {}
    
    def process_cross_worker(to_dim, answers):
        worker_class = get_worker_class(to_dim)
        system_content = worker_class.render_prompt(
            inquiry=state["inquiry"], 
            additional_context=json.dumps(answers))
        structured_llm = llm.with_structured_output(worker_class.output_schema)
        response = structured_llm.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=HUMAN_PROMPT)
        ])
        return to_dim, response

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_cross_worker, to_dim, answers): to_dim for to_dim, answers in new_inputs.items()}
        for future in futures:
            to_dim = futures[future]
            _, response = future.result()
            results[to_dim] = response
    
    # merge previous answers
    previous = state.get("worker_replies", {})
    deactivated = list(state["deactivated_workers"])
    
    def process_merge(dim, answer):
        if dim not in previous:
            return dim, answer, None
        merger_content = InquiryReplyMerger.render_prompt(
            previous_reply=previous[dim], current_reply=answer)
        merger_llm = llm.with_structured_output(InquiryReplyMerger.output_schema)
        merged_reply = merger_llm.invoke([
            SystemMessage(content=merger_content),
            HumanMessage(content=MERGER_PROMPT)])
        
        # metric check to save or deactivate
        prev_metric = calculate_worker_metric(previous[dim])
        new_metric = calculate_worker_metric(merged_reply)
        if new_metric <= prev_metric:
            return dim, merged_reply, None
        else:
            return dim, answer, dim

    with ThreadPoolExecutor() as executor:
        merge_futures = [executor.submit(process_merge, dim, answer) for dim, answer in results.items()]
        results = {}
        for future in merge_futures:
            dim, merged_reply, deactivated_dim = future.result()
            if deactivated_dim:
                deactivated.append(deactivated_dim)
            if merged_reply is not None:
                results[dim] = merged_reply

    # done
    return {
        "worker_replies": results,
        "loop_count": state["loop_count"] + 1,
        "stop": state["loop_count"] >= 2,
        "deactivated_workers": deactivated
    }


def summarizer_node(state: AgentState):
    inquiry = state["inquiry"]
    worker_replies = state.get("worker_replies", {})

    system_content = InquirySummary.render_prompt(inquiry, worker_replies)

    response = llm.invoke(
        [
            SystemMessage(content=system_content),
            HumanMessage(content="Please provide the final synthesized summary."),
        ]
    )

    return {"summary": response.content}



# --- 3. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("init_node", init_node)
workflow.add_node("prelim_nodes", prelim_nodes)  # input layer: 1 input str to all X workers
workflow.add_node("cross_nodes", cross_nodes)  # hidden layer: 1..X workers to 1..X workers
workflow.add_node("summarizer", summarizer_node)  # output layer: X workers to 1 output str

# Routing/Edges
workflow.add_edge(START, "init_node")
workflow.add_edge("init_node", "prelim_nodes")
workflow.add_edge("prelim_nodes", "cross_nodes")
# workflow.add_edge("cross_nodes", "summarizer")
workflow.add_conditional_edges(
    "cross_nodes", 
    lambda x: x["stop"],
    {
        False: "cross_nodes",
        True: "summarizer",
    }
)
workflow.add_edge("summarizer", END)

# Initialize in-memory checkpointer
checkpointer = InMemorySaver()
# Compile
graph = workflow.compile(checkpointer=checkpointer)
