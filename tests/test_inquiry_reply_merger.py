import pytest
from src.agents.workers.inquiry_reply_merger import InquiryReplyMerger
from src.agents.workers.inquiry_base import WorkerReply, AnswerItem

def test_inquiry_reply_merger_render_prompt():
    ans1 = AnswerItem(answer_type="FACTUAL", answer="A1", score=0.8)
    reply1 = WorkerReply(
        answers_list=[ans1],
        similarity_scores=[],
        connections_list=[]
    )
    
    ans2 = AnswerItem(answer_type="CAUSAL", answer="A2", score=0.9)
    reply2 = WorkerReply(
        answers_list=[ans2],
        similarity_scores=[],
        connections_list=[]
    )
    
    prompt = InquiryReplyMerger.render_prompt(reply1, reply2)
    
    assert "A1" in prompt
    assert "A2" in prompt
    assert "data1:" in prompt
    assert "data2:" in prompt
    assert "5" in prompt
