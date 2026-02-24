import math
from pydantic import BaseModel
from src.agents.workers.inquiry_base import calculate_worker_metric, WorkerReply


class InquirySupervisor:
    name = "inquiry_supervisor"

    @staticmethod
    def calculate_metric(workers_replies: dict[str, WorkerReply]) -> float:
        num_with_answer = sum(
            1 for reply in workers_replies.values() if len(reply.answers_list) > 0
        )
        sum_metrics = sum(
            calculate_worker_metric(reply) for reply in workers_replies.values()
        )
        return math.sqrt(num_with_answer) + sum_metrics
