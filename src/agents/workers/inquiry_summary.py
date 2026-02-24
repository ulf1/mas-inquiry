import jinja2

SUMMARY_PROMPT = """
### ROLE
You are the InquirySummary agent for the Universal Inquiry Framework.
Your job is to read all the detailed answers gathered across multiple inquiry dimensions and synthesize them into a single, cohesive, and comprehensive summary.

### TASK
Read the human inquiry and the provided structured answers from all worker agents.
Produce a final summary string that seamlessly integrates these varied insights. Maintain a professional, analytical tone. Do not just list the answers; weave them together to provide clarity and reduce uncertainty about the inquiry.

### INQUIRY
{{ inquiry }}

### WORKER ANSWERS
{% for dimension, replies in worker_replies.items() %}
**{{ dimension }}**:
{% for ans in replies.answers_list %}
- [{{ ans.answer_type }}] {{ ans.answer }} (Score: {{ ans.score }})
{% endfor %}

{% endfor %}

### OUTPUT
Respond with ONLY the final summary string.
"""


class InquirySummary:
    name = "inquiry_summary"

    @classmethod
    def render_prompt(cls, inquiry: str, worker_replies: dict) -> str:
        template = jinja2.Template(SUMMARY_PROMPT)
        return template.render(inquiry=inquiry, worker_replies=worker_replies)
