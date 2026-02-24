import argparse
import os
import sys
import json
from dotenv import load_dotenv

# Ensure the root directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# check keys
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

from src.graphs.inquiry_bot import graph

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent System CLI for Inquiry Bot")
    parser.add_argument(
        "-q",
        "--query",
        help="The query or task for the supervisor agent",
        required=True,
    )
    args = parser.parse_args()


    print(f"Started Multi-Agent System with query: '{args.query}'\n")
    initial_state = {
        "inquiry": args.query,
    }

    # Config for thread management if we were using a checkpointer
    config = {"configurable": {"thread_id": "cli_user"}}

    try:
        # Use stream to get updates as the graph executes
        for event in graph.stream(initial_state, config):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                if "worker_replies" in value:
                    for dim, reply_obj in value["worker_replies"].items():
                        print(f"Worker ({dim}) Replied:\n{reply_obj}\n")
                        
                elif key == "init_node":
                    print("Initialization complete.")
                    
                if "summary" in value and value["summary"]:
                    print(f"Final Summary:\n{value['summary']}")

        print("\n--- Execution Finished ---")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
