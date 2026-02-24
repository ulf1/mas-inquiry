import math
import jinja2
from pydantic import BaseModel, Field
from typing import ClassVar, List


class AnswerItem(BaseModel):
    answer: str = Field(description="The short precise answer.")
    answer_type: str = Field(description="The answer type.")
    score: float = Field(description="Relevance score for the inquiry (0.0 to 1.0).")


class SimilarityScore(BaseModel):
    i: int = Field(description="Index of first answer in the list answers_list (0 to N-1).")
    j: int = Field(description="Index of second answer in the list answers_list (0 to N-1).")
    score: float = Field(description="Similarity score (0.0 to 1.0).")


class DimensionConnection(BaseModel):
    i: int = Field(description="Index of the answer in the list answers_list (0 to N-1).")
    dimension_name: str = Field(description="Name of the related dimension.")


class WorkerReply(BaseModel):
    answers_list: list[AnswerItem] = Field(description="List of answers and relevance scores.")
    similarity_scores: list[SimilarityScore] = Field(description="Similarity scores between answers.")
    connections_list: list[DimensionConnection] = Field(
        description="Connections to related dimensions."
    )


def calculate_worker_metric(reply: WorkerReply) -> float:
    # number of found answers
    a_len = len(reply.answers_list)
    if a_len == 0:
        return 0.0

    # square root of number of answers
    term1 = math.sqrt(a_len)

    # average relevance score of answers
    mean_relevance = sum(item.score for item in reply.answers_list) / a_len

    # average similarity score of answers
    b_len = len(reply.similarity_scores)
    if b_len > 0:
        mean_similarity = sum(item.score for item in reply.similarity_scores) / b_len
    else:
        mean_similarity = 0.0

    # average number of connections per answer
    avg_connections = len(reply.connections_list) / a_len

    return term1 + mean_relevance - mean_similarity + avg_connections


ALL_DIMENSIONS = [
    "Content",
    "Agent",
    "Temporal",
    "Spatial",
    "Causal",
    "Procedural",
    "Quantitative",
    "Selective",
    "Impact",
    "Actionable",
    "Auditory",
    "Visual",
    "Kinetic",
    "Tactile",
    "Measured",
    "Affective",
    "The Void",
    "Perspective",
    "Hypothetical",
    "Ethical",
    "Probabilistic",
    "Subtext",
]


BASE_PROMPT_TEMPLATE = """
### ROLE
You are an expert worker agent for the "{{ dimension }}" dimension in the Universal Inquiry Framework.
Primary Focus: {{ primary_focus }}
Contextual Utility: {{ contextual_utility }}
Valid Answer Types: {{ answer_types }}, other
Valid Connection Dimensions: {{ valid_uif_dimensions }}

### TASK
Process the following human inquiry step by step:

0) Given an empty answer list with a capacity of up to {{ max_answers }} answers. Set a counter variable loops=0.

1) Identify dissimilar, varied, and unique answers to fill up the answer list, that are most relevant to the inquiry based on your dimension's focus. For each new answer of the answer list:
    - compute the relevance score between the answer and inquiry given the "{{ dimension }}" dimension. Use techniques like embedding model and cosine similarity.
	- identify the answer type; label answer type as "other" if the given labels don't fit.

2) Compute the similarity score between answers that have the same answer type. Similarity scores have values between 0.0 (not similar) and 1.0 (perfect match). Use techniques like Joint Sequence Attention.

3) For each answer type, select the two answers X and Y (of the same answer type) with the highest similarity score. If the similarity score is greater than {{ similarity_threshold }}, then merge both answers to merged answer Z, e.g. "[combined answer], e.g. [variant 1, variant 2]", and compute a relevance score of the merge answer Z. If the relevance score of the merged answer Z is higher than those of answer X and Y, then drop answer X and Y from the answer list, and add the merged answer Z to the answer list, and drop the similarity score between X and Y.
   
4) Set loops+=1. If loops>={{ max_fillups }}, then go to step 6.

5) If length of the answer list is lower than {{ max_answers }}, then repeat step 3.

6) For each answer, propose up to {{ max_connections }} related to other inquiry dimensions (only if there is a related dimension). Use exactly the Dimension names defined in the Universal Inquiry Framework.

### INQUIRY
{{ inquiry }}
{% if additional_context %}

### ADDITIONAL CONTEXT (Refinement)
{{ additional_context }}
{% endif %}

### OUTPUT
Respond strictly with valid JSON conforming to the schema of answers_list, similarity_scores, and connections_list.
"""


class BaseInquiryWorker:
    # worker metadata
    name: ClassVar[str] = ""
    dimension: ClassVar[str] = ""
    primary_focus: ClassVar[str] = ""
    answer_types: List[str] = []
    contextual_utility: ClassVar[str] = ""

    output_schema: ClassVar[type[BaseModel]] = WorkerReply

    @classmethod
    def render_prompt(
        cls, inquiry: str, 
        additional_context: str = "", 
        max_answers: int = 5,
        max_fillups: int = 2,
        max_connections: int = 3,
        similarity_threshold: float = 0.8,
    ) -> str:
        template = jinja2.Template(BASE_PROMPT_TEMPLATE)
        valid_uif_dimensions = [d for d in ALL_DIMENSIONS if d != cls.dimension]
        return template.render(
            dimension=cls.dimension,
            primary_focus=cls.primary_focus,
            answer_types=", ".join(cls.answer_types),
            valid_uif_dimensions=", ".join(valid_uif_dimensions),
            contextual_utility=cls.contextual_utility,
            inquiry=inquiry,
            additional_context=additional_context,
            # parameters
            max_answers=max_answers,
            max_fillups=max_fillups,
            max_connections=max_connections,
            similarity_threshold=similarity_threshold
        )


# This is a kind of residual dimensions that the LLM hallucinates when it doesn't fit the given dimensions
class InquiryOther(BaseInquiryWorker):
    name = "inquiry_other"
    dimension = "Other"
    primary_focus = "Other"
    answer_types = ["other"]
    contextual_utility = (
        "The residual dimension that is not covered by any other dimension."
    )

    @classmethod
    def set_defintion(cls, dimension: str) -> None:
        # keep dimension name as is
        cls.name = f"inquiry_{dimension.lower()}"
        cls.primary_focus = f"{dimension} dimension"
        cls.answer_types = [f"{dimension.lower()}"]
        cls.contextual_utility = f"The {dimension} dimension."


class InquiryActionable(BaseInquiryWorker):
    name = "inquiry_actionable"
    dimension = "Actionable"
    primary_focus = "Iteration"
    answer_types = ["action", "recommendation", "follow-up"]
    contextual_utility = (
        "The Vector: Points toward the necessary response or next phase."
    )


class InquiryAffective(BaseInquiryWorker):
    name = "inquiry_affective"
    dimension = "Affective"
    primary_focus = "Emotional State"
    answer_types = ["valence", "arousal", "sentiment", "vibe"]
    contextual_utility = (
        "The Atmosphere: Gauges the emotional resonance or psychological tone."
    )

class InquiryAgent(BaseInquiryWorker):
    name = "inquiry_agent"
    dimension = "Agent"
    primary_focus = "Actors & Roles"
    answer_types = ["person", "animal", "stakeholder", "victim"]
    contextual_utility = (
        "The Identity: Assigns agency, responsibility, and human impact."
    )


class InquiryAuditory(BaseInquiryWorker):
    name = "inquiry_auditory"
    dimension = "Auditory"
    primary_focus = "Sonic Signature"
    answer_types = ["pitch", "timbre", "volume", "resonance"]
    contextual_utility = (
        "The Signal: Analyzes acoustic data for stress, health, or identification."
    )

class InquiryCausal(BaseInquiryWorker):
    name = "inquiry_causal"
    dimension = "Causal"
    primary_focus = "Motivation & Logic"
    answer_types = ["intent", "cause", "rationale", "justification"]
    contextual_utility = (
        "The Catalyst: Uncovers the underlying engine driving the behavior."
    )

class InquiryContent(BaseInquiryWorker):
    name = "inquiry_content"
    dimension = "Content"
    primary_focus = "Essence & Identity"
    answer_types = ["object", "entity", "definition", "category"]
    contextual_utility = (
        "The Anchor: Establishes the baseline facts and subject matter."
    )

class InquiryEthical(BaseInquiryWorker):
    name = "inquiry_ethical"
    dimension = "Ethical"
    primary_focus = "Moral Value"
    answer_types = ["equity", "privacy", "consent", "integrity"]
    contextual_utility = (
        "The Conscience: Filters the inquiry through legal and moral constraints."
    )

class InquiryHypothetical(BaseInquiryWorker):
    name = "inquiry_hypothetical"
    dimension = "Hypothetical"
    primary_focus = "Counterfactual"
    answer_types = ["simulation", "alternative", "prediction"]
    contextual_utility = (
        "The Divergence: Explores potential realities and stress-tests outcomes."
    )

class InquiryImpact(BaseInquiryWorker):
    name = "inquiry_impact"
    dimension = "Impact"
    primary_focus = "Consequence"
    answer_types = ["implication", "significance", "value", "risk"]
    contextual_utility = (
        "The Gravity: Determines the meaningfulness or severity of the data."
    )

class InquiryKinetic(BaseInquiryWorker):
    name = "inquiry_kinetic"
    dimension = "Kinetic"
    primary_focus = "Dynamic State"
    answer_types = ["velocity", "trajectory", "fluidity", "blur"]
    contextual_utility = (
        "The Momentum: Tracks change-over-time within a single frame or event."
    )

class InquiryMeasured(BaseInquiryWorker):
    name = "inquiry_measured"
    dimension = "Measured"
    primary_focus = "Data Telemetry"
    answer_types = ["voltage", "frequency", "latency", "variance"]
    contextual_utility = (
        "The Precision: Uses objective instrumentation to bypass human bias."
    )

class InquiryPerspective(BaseInquiryWorker):
    name = "inquiry_perspective"
    dimension = "Perspective"
    primary_focus = "Bias & Lens"
    answer_types = ["framing", "subjectivity", "cultural filter"]
    contextual_utility = (
        "The Lens: Corrects for the observer effect and subjective distortion."
    )

class InquiryProbabilistic(BaseInquiryWorker):
    name = "inquiry_probabilistic"
    dimension = "Probabilistic"
    primary_focus = "Certainty"
    answer_types = ["confidence level", "risk", "entropy"]
    contextual_utility = "The Odds: Quantifies the reliability of the entire data set."

class InquiryProcedural(BaseInquiryWorker):
    name = "inquiry_procedural"
    dimension = "Procedural"
    primary_focus = "Method & Flow"
    answer_types = ["mechanism", "process", "instrument", "manner"]
    contextual_utility = (
        "The Mechanics: Details the specific steps taken to achieve the result."
    )

class InquiryQuantitative(BaseInquiryWorker):
    name = "inquiry_quantitative"
    dimension = "Quantitative"
    primary_focus = "Magnitude"
    answer_types = ["scale", "mass", "budget", "threshold", "count"]
    contextual_utility = "The Scale: Measures the volume or intensity of the subject."

class InquirySelective(BaseInquiryWorker):
    name = "inquiry_selective"
    dimension = "Selective"
    primary_focus = "Distinction"
    answer_types = ["selection", "preference", "criteria", "priority"]
    contextual_utility = "The Choice: Identifies why one path was taken over another."

class InquirySpatial(BaseInquiryWorker):
    name = "inquiry_spatial"
    dimension = "Spatial"
    primary_focus = "Location & Setting"
    answer_types = ["context", "environment", "coordinate", "layout"]
    contextual_utility = (
        "The Perimeter: Defines the physical or digital boundaries of the event."
    )

class InquirySubtext(BaseInquiryWorker):
    name = "inquiry_subtext"
    dimension = "Subtext"
    primary_focus = "Latent Meaning"
    answer_types = ["innuendo", "symbology", "unspoken rule"]
    contextual_utility = (
        "The Undercurrent: Decodes the symbolic or hidden meta-narrative."
    )

class InquiryTactile(BaseInquiryWorker):
    name = "inquiry_tactile"
    dimension = "Tactile"
    primary_focus = "Physical Surface"
    answer_types = ["pressure", "temperature", "friction", "weight"]
    contextual_utility = (
        "The Haptic: Evaluates the physical reality and material touch of data."
    )

class InquiryTemporal(BaseInquiryWorker):
    name = "inquiry_temporal"
    dimension = "Temporal"
    primary_focus = "Timing & Flow"
    answer_types = ["time", "duration", "frequency", "era", "pace"]
    contextual_utility = (
        "The Chronology: Maps the event onto a timeline to find patterns."
    )

class InquiryTheVoid(BaseInquiryWorker):
    name = "inquiry_the__void"
    dimension = "The Void"
    primary_focus = "Omission"
    answer_types = ["Absence", "silence", "blind spot", "gap"]
    contextual_utility = (
        "The Negative Space: Essential for forensics; looks for what should be there."
    )

class InquiryVisual(BaseInquiryWorker):
    name = "inquiry_visual"
    dimension = "Visual"
    primary_focus = "Appearance"
    answer_types = ["color", "shape", "texture", "composition"]
    contextual_utility = (
        "The Aesthetic: Decodes visual hierarchies and surface-level truths."
    )
