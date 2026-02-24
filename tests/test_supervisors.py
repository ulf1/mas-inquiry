import pytest
from src.agents.supervisors.inquiry_supervisor import InquirySupervisor
from src.agents.workers.inquiry_base import WorkerReply, AnswerItem, DimensionConnection

def test_calculate_metric_empty():
    replies = {}
    metric = InquirySupervisor.calculate_metric(replies)
    assert metric == 0.0

def test_calculate_metric_single():
    ans1 = AnswerItem(answer_type="FACTUAL", answer="Test", score=0.8)
    reply1 = WorkerReply(
        answers_list=[ans1],
        similarity_scores=[],
        connections_list=[]
    )
    replies = {"dim1": reply1}
    
    # Supervisor metric: sqrt(1) + worker_metric = 1.0 + (1.0 + 0.8) = 2.8
    metric = InquirySupervisor.calculate_metric(replies)
    assert round(metric, 2) == 2.8

def test_calculate_metric_multiple():
    ans1 = AnswerItem(answer_type="FACTUAL", answer="Test 1", score=1.0)
    ans2 = AnswerItem(answer_type="CAUSAL", answer="Test 2", score=0.6)
    
    reply1 = WorkerReply(
        answers_list=[ans1],
        similarity_scores=[],
        connections_list=[]
    )
    reply2 = WorkerReply(
        answers_list=[ans2],
        similarity_scores=[],
        connections_list=[]
    )
    replies = {"dim1": reply1, "dim2": reply2}
    
    metric = InquirySupervisor.calculate_metric(replies)
    
    # sqrt(2) + sum_metrics(1.0 + 1.0 + 1.0 + 0.6) = 1.4142135623730951 + 3.6
    expected = 1.4142135623730951 + 3.6
    assert round(metric, 5) == round(expected, 5)
