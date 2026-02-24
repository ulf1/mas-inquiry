import pytest
from src.agents.workers.inquiry_summary import InquirySummary
from src.agents.workers.inquiry_base import WorkerReply, AnswerItem

def test_inquiry_summary_render_prompt():
    inquiry = "What is the meaning of life?"
    
    ans1 = AnswerItem(answer_type="FACTUAL", answer="42", score=1.0)
    reply1 = WorkerReply(
        answers_list=[ans1],
        similarity_scores=[],
        connections_list=[]
    )
    worker_replies = {"dimension_x": reply1}
    
    prompt = InquirySummary.render_prompt(inquiry, worker_replies)
    
    assert inquiry in prompt
    assert "dimension_x" in prompt
    assert "42" in prompt
    assert "FACTUAL" in prompt
