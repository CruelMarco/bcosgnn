# Master's Thesis Outline: B-cos GNNs: Faithful Explanations through Dynamic Linearity

## Abstract
* A concise summary of the thesis: introducing B-cos GNNs, the concept of dynamic linearity in graphs, empirical results demonstrating state-of-the-art faithfulness, and any novel extensions explored during your master's research.

## Chapter 1: Introduction
* **1.1 Motivation:** The rise of Graph Neural Networks (GNNs) in critical applications (chemistry, healthcare, physics) and the urgent need for trustworthy, explainable predictions in high-stakes domains.
* **1.2 The Explainability Challenge:** The limitations of current post-hoc explainers (fragility, lack of faithfulness, high computational optimization cost) and the architectural complexity of existing inherently explainable models.
* **1.3 Problem Statement:** How can we design a GNN that is inherently explainable by construction, yields mathematically faithful explanations, and operates efficiently without sacrificing significant predictive power?
* **1.4 Thesis Contributions:** 
    * Identifying design principles for dynamic linearity in GNNs.
    * Instantiating B-cos GIN and GINE architectures.
    * Demonstrating state-of-the-art faithful explanations empirically.
    * Exploring architectural extensions (e.g., mean aggregation).
* **1.5 Thesis Structure:** Brief overview of the upcoming chapters.

## Chapter 2: Background and Related Work
* **2.1 Graph Neural Networks:** Overview of message passing formulation, Graph Isomorphism Networks (GIN), and handling edge attributes (GINE).
* **2.2 Explainability in GNNs:** 
    * Post-hoc methods: Gradient-based (Saliency, Integrated Gradients) and Perturbation-based (GNNExplainer, PGExplainer).
    * Inherently interpretable models (GSAT, DIR, IB-subgraph).
* **2.3 The Concept of Faithfulness:** Defining mathematically faithful explanations vs. merely plausible (human-aligned) explanations.
* **2.4 B-cos Networks:** Introduction to the B-cos transform, dynamic linearity, and weight-input alignment previously established in computer vision and NLP.

## Chapter 3: Methodology: B-cos Graph Neural Networks
* **3.1 Dynamic Linearity for Graphs:** Defining the requirements for a GNN to inherit dynamic linearity (composing dynamically linear transforms with linear aggregation).
* **3.2 Preserving Linearity with Sum-Aggregation:** Theoretical proofs demonstrating that sum-aggregation and linear readout preserve dynamic linearity across multiple message-passing layers.
* **3.3 Architecture Instantiation:**
    * **B-cos GIN:** Replacing standard ReLU-MLPs with B-cos transforms.
    * **B-cos GINE:** Extending the model to incorporate edge features as additive offsets while preserving linearity.
* **3.4 Generating Explanations:** How exact contribution maps are computed and translated into explanatory subgraphs in a single backward pass, eliminating the need for surrogate models.

## Chapter 4: Experimental Setup and Benchmarks
* **4.1 Evaluation Challenges in Graph XAI:** Discussing the issue of "trivial" datasets and the need for rigorous, confounding benchmarks.
* **4.2 The RINGS Framework:** Ensuring tasks strictly require structural necessity (P1) and feature-topology interdependence (P2).
* **4.3 Datasets:**
    * Datasets with verified rationales (BA-2Motif, MNIST-75sp, custom Di-Halo-Benzene).
    * Standard predictive benchmarks (PATTERN, NCI1, OGB-MolHIV).
* **4.4 Baselines and Metrics:** 
    * Baselines: GNNExplainer, Integrated Gradients, GSAT, Vanilla GIN/GINE.
    * Metrics: Jaccard@k, Node AUROC, Predictive F1-score/Accuracy.

## Chapter 5: Results and Analysis
* **5.1 Explanation Quality (Quantitative):** Comparing Jaccard@k and AUROC scores against baselines. Highlighting the superior structural alignment of B-cos GNNs on synthetic and chemical data.
* **5.2 Qualitative Analysis:** Visualizing and discussing the generated contribution maps (e.g., highlighting chemical precision in identifying isomers in Di-Halo-Benzene, and topological isolation in BA-2Motif).
* **5.3 Predictive Performance Trade-offs:** Analyzing the (modest) cost of inherent interpretability on predictive accuracy across standard benchmarks.
* **5.4 Sensitivity Analysis:** The effect of the alignment pressure hyperparameter ($B$) on both predictive accuracy and explanation AUROC.
* **5.5 Computational Efficiency:** Benchmarking the inference time for generating explanations against post-hoc optimization methods, showcasing orders of magnitude speedups.

## Chapter 6: Extensions and Ongoing Work
> [!NOTE]
> *Based on your active open files (e.g., `NCI1_mean_agg.54783.0.out`), it looks like you are actively exploring mean aggregation! This chapter is the perfect place to include novel thesis contributions that go beyond the NeurIPS paper.*
* **6.1 Limitations of Sum-Aggregation:** Discussing scenarios where sum-aggregation might fail or be suboptimal for certain graph topologies.
* **6.2 Exploring Mean Aggregation:** Adapting the B-cos framework for mean aggregation. Discussing the theoretical implications (does it strictly preserve dynamic linearity?) and presenting preliminary empirical results on datasets like NCI1.
* **6.3 Future Directions:** Potential for edge-level contributions (beyond node/feature level attribution) and exploring attention-based aggregators.

## Chapter 7: Conclusion
* **7.1 Summary of Findings:** Re-iterating the core successes of the B-cos GNN framework in bridging the gap between faithfulness and accuracy.
* **7.2 Broader Impact:** The societal and practical implications of deploying fast, transparent, and inherently interpretable GNNs in real-world high-stakes domains.

## References

## Appendices
* Appendix A: Detailed Theoretical Proofs (e.g., Lemma 1 and Proposition 1).
* Appendix B: Dataset Construction Details (e.g., the Di-Halo-Benzene generation process).
* Appendix C: Extended Qualitative Examples and Hyperparameter specifications.
