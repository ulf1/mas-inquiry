---
Prompt: analyse the algorithm coded in src/ and in the README.md; search for literature or research paper that implemented this algorithm before, or find research ideas with a high similarity. Also look for a AI research paper since 70s up today. save your findings in literature_search.md
Warning: I have not checked any of the sources yet. Could be all fake.
---

# Literature Review: MAS Inquiry Framework

This document maps the algorithm implemented in `src/` to prior research and related work, spanning from the 1960s to the 2020s.

---

## 1. Algorithm Summary

The **MAS Inquiry Framework** is a multi-agent system that processes human inquiries using:

1. **22 specialized "Question Workers"**, each focusing on a different interrogative dimension (Content, Causal, Temporal, Spatial, Ethical, etc.)
2. A **3-phase graph architecture** resembling a neural network:
   - **Input layer** (`prelim_nodes`): All workers process the inquiry in parallel, producing structured answers with relevance scores, similarity scores, and connections to other dimensions.
   - **Hidden layers** (`cross_nodes`, iterated up to 3 times): Workers receive answers from connected workers, merge them with previous results (skip connections), and deactivate workers whose merged metric does not improve (dynamic pruning).
   - **Output layer** (`summarizer`): All surviving answers are synthesized into a final summary.
3. A **metric function** per worker: `√(num_answers) + mean_relevance − mean_similarity + avg_connections`, used to decide whether to keep or deactivate a worker after merging.
4. An **answer merging algorithm** inside each worker prompt: iteratively find the two most similar answers (same type), merge if similarity > 0.8 and merged relevance exceeds originals, capped by max iterations.
5. The **InquiryOther** catch-all worker as a residual term, ensuring the decomposition is MECE (Mutually Exclusive, Collectively Exhaustive).

---

## 2. Classic AI & Cognitive Science Foundations (1960s–1990s)

### 2.1 Semantic Networks & Spreading Activation (1969–1975)

| | |
|---|---|
| **Key Papers** | Collins & Quillian, "Retrieval Time from Semantic Memory" (1969); Collins & Loftus, "A Spreading-Activation Theory of Semantic Processing" (1975) |
| **Core Idea** | Knowledge organized as a network of concept nodes connected by typed links. Activation spreads from a query node outward to related concepts. |
| **Similarity** | The cross-worker connections (where Worker A sends an answer to Worker B via `DimensionConnection`) are a form of spreading activation — information propagates from one semantic dimension to related dimensions through explicit links. The framework's "connections_list" is essentially a spreading activation graph where each answer activates related dimensions. |

### 2.2 The Blackboard Architecture (1971–1985)

| | |
|---|---|
| **Key Papers** | Erman, Hayes-Roth, Lesser & Reddy, "The Hearsay-II Speech-Understanding System" (1980, *Computing Surveys*); Hayes-Roth, "A Blackboard Architecture for Control" (1985, *Artificial Intelligence*) |
| **Core Idea** | Multiple independent **knowledge sources** (specialists) cooperate through a shared **blackboard** data structure. Each specialist monitors the blackboard and contributes partial solutions when its expertise applies. A control module orchestrates the process. |
| **Similarity** | **This is the closest classical match.** The MAS Inquiry Framework's `AgentState` dictionary acts as the blackboard. The 22 Question Workers are the knowledge sources/specialists. The graph workflow (init → prelim → cross → summarize) acts as the control mechanism. Workers read from and write to shared state, just like blackboard knowledge sources. The iterative refinement in `cross_nodes` mirrors Hearsay-II's iterative hypothesis refinement. |

### 2.3 Society of Mind (1986)

| | |
|---|---|
| **Key Work** | Minsky, M. *The Society of Mind* (1986, Simon & Schuster) |
| **Core Idea** | Intelligence emerges from the interactions of many simple, specialized agents — none intelligent alone, but collectively capable of complex cognition. |
| **Similarity** | The framework's design philosophy directly echoes Minsky: 22 individually simple workers (each just an LLM prompt with a narrow focus) collectively produce a comprehensive understanding by cooperating through shared state. No single worker "understands" the inquiry — intelligence emerges from the ensemble. |

### 2.4 Contract Net Protocol (1980)

| | |
|---|---|
| **Key Paper** | Smith, R.G. "The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver" (1980, *IEEE Transactions on Computers*) |
| **Core Idea** | A **manager** decomposes tasks and broadcasts calls for proposals to **contractor** agents. Contractors bid, manager selects the best, and contracts are awarded. Supports hierarchical decomposition. |
| **Similarity** | The framework's supervisor-worker pattern with the inquiry graph orchestrating task distribution to dimension workers follows the CNP model. The `init_node` acts as the manager that broadcasts to all workers. The metric-based deactivation of workers is analogous to contract evaluation — workers that don't improve the metric are essentially not re-contracted. |

### 2.5 Distributed AI & Cooperative Problem Solving (1987–1995)

| | |
|---|---|
| **Key Papers** | Durfee, Lesser & Corkill, "Coherent Cooperation Among Communicating Problem Solvers" (1987); Durfee & Lesser, "Trends in Cooperative Distributed Problem Solving" (1989); Decker & Lesser, "Designing a Family of Coordination Algorithms" (1995) |
| **Core Idea** | Multiple agents with partial information cooperate by exchanging partial solutions, using coordination mechanisms to maintain coherence. No single agent can solve the problem alone. |
| **Similarity** | Each Question Worker has only partial "knowledge" (its dimension). The `cross_nodes` phase implements cooperative problem solving — workers exchange answers through connections and refine each other's partial solutions. The merge operation ensures coherent integration of distributed partial answers. |

### 2.6 Parallel Distributed Processing / Connectionism (1986)

| | |
|---|---|
| **Key Work** | Rumelhart, McClelland & the PDP Research Group, *Parallel Distributed Processing: Explorations in the Microstructure of Cognition* (1986) |
| **Core Idea** | Cognition emerges from parallel interactions among many simple processing units. Knowledge is distributed across connections, not localized in any single unit. |
| **Similarity** | The README explicitly draws this analogy: _"It's basically like sparse connectivity neural networks."_ The framework uses parallel worker execution (`ThreadPoolExecutor`), distributed knowledge across dimension workers, and connection-based information flow — but operates on **symbolic tokens and text** rather than floating-point activations. The README's author asks: _"I wondered if all the biology analogies and neural network design ideas from the past were never supposed to just use floating point numbers."_ |

### 2.7 Mixture of Experts (1991)

| | |
|---|---|
| **Key Paper** | Jacobs, Jordan, Nowlan & Hinton, "Adaptive Mixtures of Local Experts" (1991, *Neural Computation*) |
| **Core Idea** | Multiple "expert" sub-networks, each specialized on a subset of the input space, are combined via a **gating network** that routes inputs to the appropriate expert. Reduces interference by localizing weight adjustments to responsible experts. |
| **Similarity** | **Very high similarity.** Each Question Worker is a specialized "expert" for its dimension. The framework's metric and deactivation mechanism acts as a soft gating function — experts whose merged metric doesn't improve are "gated out" (deactivated). The similarity merging within each worker reduces redundancy, much like MoE reduces interference. Modern MoE architectures (used in models like Mixtral) continue this tradition. |

### 2.8 Neural Network Pruning & Skip Connections (1989–1995)

| | |
|---|---|
| **Key Papers** | LeCun, Denker & Solla, "Optimal Brain Damage" (1989); Hassibi, Stork & Wolff, "Optimal Brain Surgeon" (1993); Hochreiter, "Untersuchungen zu dynamischen neuronalen Netzen" (1991, diploma thesis — first residual connections in RNNs); He et al., "Deep Residual Learning" (2015, popularized skip connections) |
| **Core Idea** | **Pruning**: Remove redundant connections/weights to improve generalization. **Skip (residual) connections**: Allow information to bypass layers, preventing gradient vanishing and enabling deeper networks. |
| **Similarity** | The framework implements both concepts at the agent level: **Dynamic pruning** occurs when `new_metric <= prev_metric` causes worker deactivation. **Skip connections** are implemented by merging a worker's previous state with its new output (the `InquiryReplyMerger`). The README explicitly describes these: _"Skip connections: Each Question Worker's previous state and output state are merged. Variable depth/recursion: If a skip connection doesn't improve a metric, the particular Question Worker is deactivated."_ |


---

## 3. The Bridge Era (2000–2015)

### 3.1 BDI Agent Architecture & FIPA Standards (1995–2005)

| | |
|---|---|
| **Key Papers** | Wooldridge & Jennings, "Intelligent Agents: Theory and Practice" (1995, *The Knowledge Engineering Review*); Wooldridge, *Reasoning about Rational Agents* (2000); Rao & Georgeff, "BDI Agents: From Theory to Practice" (1995, *ICMAS*) |
| **Core Idea** | Agents modeled with **Beliefs** (knowledge about the world), **Desires** (goals), and **Intentions** (committed plans). Agents communicate via standardized protocols (FIPA-ACL). The Gaia methodology provides structured agent-oriented design. |
| **Similarity** | Each Question Worker has implicit beliefs (its dimension's focus and answer types), desires (maximize the worker metric), and intentions (fill up to N answers, merge duplicates, propose connections). The structured `WorkerReply` schema acts like a standardized agent communication language — all workers speak the same schema, similar to FIPA-ACL standardizing inter-agent messages. |

### 3.2 Semantic Web, Ontologies & Knowledge Graphs (2001–2012)

| | |
|---|---|
| **Key Works** | Berners-Lee, Hendler & Lassila, "The Semantic Web" (2001, *Scientific American*); OWL becomes W3C Recommendation (2004); Google Knowledge Graph (2012) |
| **Core Idea** | Structure world knowledge using formal ontologies (OWL/RDF) with typed entities, properties, and relations. Knowledge graphs represent information as entity–relation–entity triples, enabling machine reasoning over structured knowledge. |
| **Similarity** | The framework's 22 dimensions function as a lightweight ontology for interrogative knowledge — each dimension is a "class" of question with defined "answer types" (the `answer_types` field). The `DimensionConnection` objects linking answers to dimensions are essentially knowledge graph edges connecting entities (answers) to classes (dimensions). The framework does at the agent level what OWL does at the data level: categorize and relate knowledge across typed dimensions. |

### 3.3 IBM Watson DeepQA / Jeopardy! (2011)

| | |
|---|---|
| **Key Papers** | Ferrucci et al., "Building Watson: An Overview of the DeepQA Project" (2010, *AI Magazine*); Ferrucci, "Introduction to 'This is Watson'" (2012, *IBM Journal of R&D*) |
| **Core Idea** | A massively parallel pipeline of hundreds of specialized NLP components — question classifiers, candidate generators, evidence scorers, and answer mergers — all operating via Apache UIMA. Multiple hypotheses are generated, scored, and fused to produce a final answer with a confidence score. |
| **Similarity** | **Very high structural similarity.** Watson's architecture is arguably the closest industrial-scale predecessor: (1) Question analysis decomposes the query into focus/LAT/relations → the Inquiry Framework decomposes into 22 dimensions. (2) Hundreds of parallel specialist components → 22 parallel Question Workers. (3) Candidate generation + evidence scoring → answers with relevance scores. (4) Answer merging and confidence scoring → `InquiryReplyMerger` with `calculate_worker_metric`. (5) Final answer selection → `summarizer_node`. The key difference: Watson used hundreds of traditional NLP modules, while this framework uses LLM-prompted agents. |

### 3.4 Ensemble Methods: Random Forests & Gradient Boosting (2001–2014)

| | |
|---|---|
| **Key Papers** | Breiman, "Random Forests" (2001, *Machine Learning*); Dietterich, "Ensemble Methods in Machine Learning" (2000, *MCS*); Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001); Chen & Guestrin, "XGBoost" (2016, *KDD*) |
| **Core Idea** | Combine predictions from multiple specialized models (weak learners) to produce a stronger aggregate prediction. **Bagging** (Random Forests) trains models in parallel on bootstrapped data, then averages. **Boosting** trains models sequentially, each correcting the previous one's errors. |
| **Similarity** | The framework implements an agent-level ensemble: (1) **Parallel phase** (`prelim_nodes`): all 22 workers process the query independently, like bagging. (2) **Sequential refinement** (`cross_nodes` loop): iterative re-processing with merged context from previous rounds, like boosting. (3) **Aggregation** (`summarizer`): combining all workers' outputs into a final answer, like ensemble voting. The metric-based deactivation is analogous to pruning weak learners from the ensemble. |

### 3.5 Attention Mechanism (2014–2015)

| | |
|---|---|
| **Key Papers** | Bahdanau, Cho & Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014/2015, *ICLR*); Luong, Pham & Manning, "Effective Approaches to Attention-based Neural Machine Translation" (2015) |
| **Core Idea** | Instead of compressing an entire input into a single fixed-size vector, the model dynamically attends to different parts of the input at each decoding step, computing a weighted sum of encoder hidden states. |
| **Similarity** | The `cross_nodes` mechanism is a form of attention at the agent level: each worker doesn't receive all information equally — it receives only the answers connected to it via `DimensionConnection`, which act as attention weights routing relevant information from specific source workers. The relevance score on each answer functions like an attention weight, determining how much influence each piece of information should have. |

### 3.6 Dropout & Stochastic Depth (2012–2016)

| | |
|---|---|
| **Key Papers** | Hinton et al., "Improving neural networks by preventing co-adaptation of feature detectors" (2012); Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014, *JMLR*); Huang et al., "Deep Networks with Stochastic Depth" (2016) |
| **Core Idea** | **Dropout**: randomly deactivate neurons during training to prevent co-adaptation and improve generalization. **Stochastic depth**: randomly skip entire layers (residual blocks), training an implicit ensemble of networks of varying depths. |
| **Similarity** | The framework's worker deactivation mechanism is conceptually equivalent to stochastic depth — entire "layer" nodes (workers) are deactivated when they don't improve the metric, effectively creating a dynamic-depth network. Unlike random dropout, the deactivation here is **metric-driven** rather than random, making it a deterministic form of adaptive depth. |

---

## 4. Modern LLM-Era Related Work (2020s)

### 4.1 Multi-Agent LLM Frameworks for Question Decomposition

| | |
|---|---|
| **Key Systems** | MAgICoRe (Multi-Agent, Iterative, Coarse-to-Fine Refinement); TRQA (Tree-of-Reasoning Question Decomposition); AutoGen (Microsoft); CrewAI; LangGraph multi-agent patterns |
| **Core Idea** | Specialized LLM agents (Solver, Reviewer, Refiner, Orchestrator) cooperate iteratively to decompose complex questions, generate sub-answers, refine them via feedback loops, and synthesize final results. |
| **Similarity** | The MAS Inquiry Framework is structurally similar but uses **dimension-based decomposition** instead of role-based decomposition. Where MAgICoRe has Solver→Reviewer→Refiner, the Inquiry Framework has prelim→cross→merge→summarize. The iterative loop with metric-based stopping (`loop_count >= 2`) matches the convergence criteria used in these systems. |

### 4.2 Least-to-Most Prompting & Self-Ask (2022)

| | |
|---|---|
| **Key Papers** | Zhou et al., "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models" (2022); Press et al., "Measuring and Narrowing the Compositionality Gap in Language Models" (2022, Self-Ask) |
| **Core Idea** | Decompose a complex question into simpler sub-questions, solve them sequentially, and feed answers from earlier steps into later ones. |
| **Similarity** | The Question Workers implement a parallel version of this: all dimension-specific sub-questions are generated simultaneously, cross-connected, and iteratively refined. The key difference is that the Inquiry Framework decomposes by **semantic dimension** (What? Why? How? When? ...) rather than by logical reasoning steps. |

### 4.3 Interrogative Framework / 5W1H in AI

| | |
|---|---|
| **Tradition** | The 5W1H (Who, What, When, Where, Why, How) framework is one of the oldest structuring principles in journalism, rhetoric, and knowledge acquisition (Hermagoras of Temnos, ~150 BC; Cicero, ~80 BC; Quintilian, ~95 AD). |
| **Modern AI Use** | Used in prompt engineering, information extraction, knowledge graph construction, and question answering systems. |
| **Similarity** | The framework's 22 dimensions are an **extended 5W1H** — the base 10 dimensions map directly to interrogative types (What → Content, Who → Agent, When → Temporal, Where → Spatial, Why → Causal, How → Procedural, How Much → Quantitative, Which → Selective, So What → Impact, What Next → Actionable). The additional multimodal and hidden dimensions extend this to sensory and meta-cognitive interrogatives. |

### 4.4 MECE Principle in Knowledge Decomposition

| | |
|---|---|
| **Origin** | Barbara Minto, McKinsey & Company (late 1960s); formalized in *The Pyramid Principle* (1987) |
| **Core Idea** | When decomposing a problem, categories should be **Mutually Exclusive** (no overlap) and **Collectively Exhaustive** (no gaps). |
| **Similarity** | The README explicitly references MECE. The InquiryOther worker acts as a **residual category** to ensure collective exhaustiveness — any answer that doesn't fit the 21 named dimensions is caught by InquiryOther. The similarity-based merging within workers enforces mutual exclusivity by deduplicating overlapping answers. |

---

## 5. Key Novelty & Differentiation

The MAS Inquiry Framework combines ideas from multiple research traditions in a way that, to the best of this review, **has not been published as a unified system**:

| Aspect | Classical Analogue | This Framework |
|---|---|---|
| Architecture | Blackboard (Hearsay-II) | LangGraph state graph with typed worker agents |
| Expert routing | Mixture of Experts (gating) | Metric-based dynamic activation/deactivation |
| Information flow | Spreading activation | Cross-worker `DimensionConnection` routing |
| Deduplication | Neural pruning | LLM-prompted similarity-based answer merging |
| Skip connections | Residual learning (Hochreiter 1991, He 2015) | Previous/current reply merging with metric gate |
| Decomposition | 5W1H, MECE | 22 interrogative dimensions + catch-all |
| Agents as neurons | PDP (Rumelhart 1986), Society of Mind (Minsky 1986) | LLM-backed symbolic processing units in a sparse dynamic graph |
| Pipeline QA | Watson DeepQA (2011) | LLM agents replace hundreds of NLP modules |
| Ensemble | Random Forest / Boosting | Parallel workers + iterative refinement + deactivation |
| Stopping criterion | Convergence in iterative solvers | Loop count + metric stagnation → deactivation |

The most distinctive feature is the **neural-network-like architecture operating on symbolic/linguistic data through LLM agents** — essentially a "neural network" where each "neuron" is a prompted language model rather than a differentiable function, and the "activations" are structured text answers rather than floating-point vectors.

---

## 6. Suggested Further Reading

### Classic (1960s–1990s)
1. Collins, A.M. & Quillian, M.R. (1969). "Retrieval Time from Semantic Memory." *Journal of Verbal Learning and Verbal Behavior*, 8(2), 240–247.
2. Collins, A.M. & Loftus, E.F. (1975). "A Spreading-Activation Theory of Semantic Processing." *Psychological Review*, 82(6), 407–428.
3. Smith, R.G. (1980). "The Contract Net Protocol." *IEEE Transactions on Computers*, C-29(12), 1104–1113.
4. Erman, L.D., Hayes-Roth, F., Lesser, V.R. & Reddy, D.R. (1980). "The Hearsay-II Speech Understanding System: Integrating Knowledge to Resolve Uncertainty." *Computing Surveys*, 12(2), 213–253.
5. Hayes-Roth, B. (1985). "A Blackboard Architecture for Control." *Artificial Intelligence*, 26(3), 251–321.
6. Minsky, M. (1986). *The Society of Mind*. Simon & Schuster.
7. Rumelhart, D.E. & McClelland, J.L. (1986). *Parallel Distributed Processing: Explorations in the Microstructure of Cognition*. MIT Press.
8. Minto, B. (1987). *The Pyramid Principle: Logic in Writing and Thinking*. Minto International.
9. Durfee, E.H., Lesser, V.R. & Corkill, D.D. (1987). "Coherent Cooperation Among Communicating Problem Solvers." *IEEE Transactions on Computers*, C-36(11), 1275–1291.
10. LeCun, Y., Denker, J.S. & Solla, S.A. (1989). "Optimal Brain Damage." *Advances in Neural Information Processing Systems 2 (NeurIPS)*.
11. Jacobs, R.A., Jordan, M.I., Nowlan, S.J. & Hinton, G.E. (1991). "Adaptive Mixtures of Local Experts." *Neural Computation*, 3(1), 79–87.
12. Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen*. Diploma thesis, TU München. (First proposal of residual connections for RNNs)
13. Hassibi, B., Stork, D.G. & Wolff, G.J. (1993). "Optimal Brain Surgeon and General Network Pruning." *IEEE International Conference on Neural Networks*.
14. Decker, K.S. & Lesser, V.R. (1995). "Designing a Family of Coordination Algorithms." *ICMAS-95*.
15. Wooldridge, M. & Jennings, N.R. (1995). "Intelligent Agents: Theory and Practice." *The Knowledge Engineering Review*, 10(2), 115–152.

### Bridge Era (2000–2015)
16. Dietterich, T.G. (2000). "Ensemble Methods in Machine Learning." *Multiple Classifier Systems (MCS)*, LNCS 1857.
17. Wooldridge, M. (2000). *Reasoning about Rational Agents*. MIT Press.
18. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32.
19. Berners-Lee, T., Hendler, J. & Lassila, O. (2001). "The Semantic Web." *Scientific American*, 284(5), 34–43.
20. Ferrucci, D. et al. (2010). "Building Watson: An Overview of the DeepQA Project." *AI Magazine*, 31(3), 59–79.
21. Hinton, G.E. et al. (2012). "Improving neural networks by preventing co-adaptation of feature detectors." *arXiv:1207.0580*.
22. Srivastava, N. et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*, 15, 1929–1958.
23. Bahdanau, D., Cho, K. & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR 2015*.

### Modern (2015–2025)
24. He, K., Zhang, X., Ren, S. & Sun, J. (2015). "Deep Residual Learning for Image Recognition." *arXiv:1512.03385*.
25. Huang, G. et al. (2016). "Deep Networks with Stochastic Depth." *ECCV 2016*.
26. Shazeer, N. et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *arXiv:1701.06538*.
27. Press, O. et al. (2022). "Measuring and Narrowing the Compositionality Gap in Language Models." (Self-Ask)
28. Zhou, D. et al. (2022). "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models." *arXiv:2205.10625*.
29. MAgICoRe — Multi-Agent Iterative Coarse-to-Fine Refinement (2024, OpenReview).
30. TRQA — Tree-of-Reasoning Question Decomposition (AAAI).
