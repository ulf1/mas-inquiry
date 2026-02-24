import jinja2
import json
from .inquiry_base import WorkerReply

MERGER_PROMPT = """
### ROLE
You are the InquiryReplyMerger agent. Your job is to clean up and merge multiple lists of inquiry answers, similarities, and dimension connections.

### TASK
Follow these steps strictly:

0) Set a counter variable loops=0.

1) Concat data1.answers_list and data2.answers_list. Concat data1.connections_list and data2.connections_list

2) Compute the similarity score between answers that have the same answer type. Similarity scores have values between 0.0 (not similar) and 1.0 (perfect match). Use techniques like Joint Sequence Attention.

3) For each answer type, select the two answers X and Y (of the same answer type) with the highest similarity score. If the similarity score is greater than {{ similarity_threshold }}, then merge both answers to merged answer Z, e.g. "[combined answer], e.g. [variant 1, variant 2]", and compute a relevance score of the merge answer Z. If the relevance score of the merged answer Z is higher than those of answer X and Y, then 
   - drop answer X and Y from the answer list, 
   - and add the merged answer Z to the answer list,
   - and drop the similarity score between X and Y,
   - drop connections of X and Y from the connection list,
   - add connection of Z to connection list.

4) Set loops+=1. If loops>={{ max_removals }}, then go to step 6.

5) If length of the answer list is greater than {{ max_answers }}, then repeat step 3.

6) Stop.

### INPUT DATA
data1: {{ data1 }}

data2: {{ data2 }}

### OUTPUT
Respond strictly with valid JSON conforming to the schema of answers_list, similarity_scores, and connections_list.
"""


class InquiryReplyMerger:
    name = "inquiry_reply_merger"
    output_schema = WorkerReply

    @classmethod
    def render_prompt(cls, previous_reply: WorkerReply, current_reply: WorkerReply) -> str:
        num1, num2 = len(previous_reply.answers_list), len(current_reply.answers_list)
        # run merge query
        template = jinja2.Template(MERGER_PROMPT)
        return template.render(
            data1=previous_reply.model_dump_json(), 
            data2=current_reply.model_dump_json(), 
            max_answers=5,
            max_removals=max(0, num1 + num2 - 5),
            similarity_threshold=0.8
        )
