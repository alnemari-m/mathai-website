# Reading & Review Papers

Essential papers covering the mathematical foundations of machine learning and AI.

---

## Assignment Requirements

**Component Weight:** 10% of final grade

**Requirements:**

- Read and review the **5 required papers** below throughout the semester
- Submit written reviews (2-3 pages each)
- Focus on mathematical concepts and techniques
- Connect paper content to course topics

**Review Structure:**

1. Summary of main mathematical concepts
2. Key theorems, proofs, or derivations
3. Connections to course material
4. Critical analysis of mathematical approach
5. Personal insights and questions

---

## BONUS OPPORTUNITY

**Want to go deeper?** Read additional papers from the Optional Papers section below!

- Discuss any optional paper with me during office hours
- Gain deeper understanding of mathematical foundations
- **Earn bonus points** towards your final grade
- **More papers = deeper expertise in Mathematics of AI**

The more you read, the stronger your mathematical foundation will become. Students who engage with optional papers consistently demonstrate superior understanding and research capability.

---

## REQUIRED PAPERS (5 Papers - Must Read)

**All papers are available on arXiv and are ≤20 pages**

---

### Required Paper 1: Linear Algebra & Matrix Decomposition

#### **A Tutorial on Principal Component Analysis**
**Authors:** Shlens (2014)
**Pages:** 12 pages
**arXiv:** [1404.1100](https://arxiv.org/abs/1404.1100)

**Topics Covered:**

- Covariance matrices and their eigenstructure (Topics 1-2)
- Eigendecomposition and Singular Value Decomposition (Topic 3)
- Variance maximization and projection (Topic 2)
- Dimensionality reduction from first principles (Application 2)

**Why This Paper is Required:**
This tutorial derives PCA step-by-step from first principles, demonstrating exactly how linear algebra (Topics 1-3) powers one of the most important tools in data science. You'll see eigenvalues, eigenvectors, covariance matrices, and SVD working together in a concrete application. The paper assumes only basic linear algebra, making it the perfect bridge between course theory and practice.

**What Makes It Excellent:**

- Short, clear, and self-contained
- Complete mathematical derivation from scratch
- Connects abstract linear algebra to real data analysis
- Directly supports Application 2 (Dimensionality Reduction)

**Course Topics:** Topics 1, 2, 3, Application 2

---

### Required Paper 2: Optimization Theory

#### **An Overview of Gradient Descent Optimization Algorithms**
**Authors:** Ruder (2017)
**Pages:** 14 pages
**arXiv:** [1609.04747](https://arxiv.org/abs/1609.04747)

**Topics Covered:**

- Gradient descent and its variants (Topic 6)
- Momentum, Nesterov accelerated gradient
- Adaptive learning rate methods (Adagrad, RMSprop, Adam)
- Challenges in optimization: saddle points, local minima

**Why This Paper is Required:**
This paper provides a clear, accessible overview of all the major gradient descent optimization algorithms used in modern deep learning. It explains the mathematical formulation of each optimizer, compares their properties, and discusses when to use each one. Understanding these optimizers is essential for training any neural network.

**What Makes It Excellent:**

- Covers every major optimizer in one place
- Clear mathematical formulations with intuitive explanations
- Practical guidance on choosing optimizers
- One of the most cited optimization overviews in deep learning

**Course Topics:** Topics 4, 6, Application 1

---

### Required Paper 3: Probability & Generative Models

#### **Auto-Encoding Variational Bayes**
**Authors:** Kingma, Welling (2014)
**Pages:** 14 pages
**arXiv:** [1312.6114](https://arxiv.org/abs/1312.6114)

**Topics Covered:**

- Probability distributions and latent variables (Topic 5)
- Variational inference and the ELBO
- Reparameterization trick (Topics 4 + 5)
- Gaussian distributions in generative models

**Why This Paper is Required:**
The VAE paper is a landmark that elegantly combines probability theory (Topic 5) with optimization (Topic 6) and calculus (Topic 4). It shows how to learn probability distributions from data using neural networks. The mathematical derivation of the Evidence Lower Bound (ELBO) is a masterclass in applying Bayes' theorem, KL divergence, and expectations — all key concepts from Topic 5.

**What Makes It Excellent:**

- Foundational paper for modern generative AI
- Elegant mathematical derivation using course concepts
- Demonstrates probability + optimization working together
- Directly connects to Application 3 (Density Estimation)

**Course Topics:** Topics 4, 5, 6, Application 3

---

### Required Paper 4: Gradient-Based Learning

#### **Adam: A Method for Stochastic Optimization**
**Authors:** Kingma, Ba (2015)
**Pages:** 15 pages
**arXiv:** [1412.6980](https://arxiv.org/abs/1412.6980)

**Topics Covered:**

- First and second moment estimation of gradients (Topics 4 + 5)
- Bias correction in running averages
- Convergence analysis for convex objectives (Topic 6)
- Gradient computation and adaptive learning rates

**Why This Paper is Required:**
Adam is the most widely used optimizer in deep learning. This paper derives it mathematically, showing how gradient computation (Topic 4), expected values and variance (Topic 5), and optimization theory (Topic 6) come together. The convergence proof provides a rigorous example of mathematical analysis applied to a practical algorithm. Every ML practitioner should understand this derivation.

**What Makes It Excellent:**

- The default optimizer in most deep learning frameworks
- Clean mathematical derivation with convergence proof
- Combines concepts from calculus, probability, and optimization
- Directly applicable to all neural network training

**Course Topics:** Topics 4, 5, 6

---

### Required Paper 5: Kernel Methods & SVM

#### **Learning Theory and Support Vector Machines - A Primer**
**Authors:** Banf (2019)
**Pages:** ~15 pages
**arXiv:** [1902.04622](https://arxiv.org/abs/1902.04622)

**Topics Covered:**

- Statistical learning theory fundamentals
- Empirical vs structural risk minimization
- Support Vector Machines (Application 4)
- Kernel trick and feature spaces (Topic 2)

**Why This Paper is Required:**
This primer introduces statistical learning theory and SVMs in an accessible way. It covers the mathematical foundations of why learning from data works (risk minimization), then shows how SVMs use geometric concepts (margins, hyperplanes) from our course to build powerful classifiers. It directly supports Application 4 and ties together linear algebra (Topics 1-2) and optimization (Topic 6).

**What Makes It Excellent:**

- Accessible introduction to learning theory
- Connects mathematical foundations to practical classification
- Covers both theory (risk minimization) and practice (SVM)
- Short and focused — ideal for course-level reading

**Course Topics:** Topics 1, 2, 6, Application 4

---

## Why These 5 Papers?

**Comprehensive Coverage:**

- **Paper 1** covers linear algebra and matrix decomposition (Topics 1-3)
- **Paper 2** covers optimization algorithms (Topic 6)
- **Paper 3** covers probability and generative models (Topic 5)
- **Paper 4** covers gradient computation and adaptive methods (Topic 4)
- **Paper 5** synthesizes multiple topics in SVM classification (Application 4)

**Student-Friendly:**

- All papers are ≤20 pages
- Accessible mathematical rigor without being overwhelming
- Build progressively on course material
- Mix of classical foundations and modern perspectives

**Practical Relevance:**

- Each paper connects directly to ML practice
- Cover the three main course applications (PCA, GMM, SVM)
- Provide foundation for understanding modern deep learning

---

## Suggested Reading Schedule

| Week | Paper | Topic | Review Due |
|------|-------|-------|------------|
| **3-4** | Paper 1 (PCA Tutorial) | Topics 1-3 | Week 5 |
| **5-6** | Paper 4 (Adam Optimizer) | Topic 4 | Week 7 |
| **7-8** | Paper 2 (Gradient Descent Overview) | Topic 6 | Week 9 |
| **9-10** | Paper 3 (VAE) | Topic 5 | Week 11 |
| **11-12** | Paper 5 (SVM Primer) | Application 4 | Week 13 |

---

## OPTIONAL PAPERS (15 Papers - For Bonus)

Want to earn bonus points and gain deeper understanding? Read any of these papers and discuss them with me during office hours!

**All papers available on arXiv and are ≤20 pages**

---

### Linear Algebra & Matrix Theory (4 papers)

#### **Optional 1: Linear Algebra in Transformers**
**Title:** "Attention Is All You Need"
**Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017) | **Pages:** 15 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
**Focus:** Matrix multiplications, linear projections, scaled dot-product attention
**Why Read:** See how linear algebra (matrix operations, projections, inner products) powers the architecture behind ChatGPT and modern AI

#### **Optional 2: Matrix Factorization in Recommender Systems**
**Title:** "Neural Collaborative Filtering"
**Authors:** He, Liao, Zhang, Nie, Hu, Chua (2017) | **Pages:** 10 | [arXiv:1708.05031](https://arxiv.org/abs/1708.05031)
**Focus:** Matrix factorization extended with neural networks for recommendation
**Why Read:** Shows how matrix decomposition concepts (Topic 3) are applied in real recommendation systems

#### **Optional 3: Vector Spaces in NLP**
**Title:** "Efficient Estimation of Word Representations in Vector Space"
**Authors:** Mikolov, Chen, Corrado, Dean (2013) | **Pages:** 12 | [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
**Focus:** Word embeddings as vectors, linear relationships in semantic space
**Why Read:** Demonstrates that words can be represented as vectors where linear algebra operations capture meaning (king - man + woman = queen)

#### **Optional 4: Normalization and Matrix Statistics**
**Title:** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
**Authors:** Ioffe, Szegedy (2015) | **Pages:** 11 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
**Focus:** Statistics of activations, normalization transformations, gradient flow
**Why Read:** Shows how matrix statistics (mean, variance) and normalization improve neural network training

---

### Optimization Methods (4 papers)

#### **Optional 5: Escaping Saddle Points**
**Title:** "How to Escape Saddle Points Efficiently"
**Authors:** Jin, Ge, Netrapalli, Kakade, Jordan (2017) | **Pages:** 9 | [arXiv:1703.00887](https://arxiv.org/abs/1703.00887)
**Focus:** Non-convex optimization, perturbed gradient descent, convergence to second-order stationary points
**Why Read:** Explains mathematically why gradient descent succeeds in non-convex deep learning landscapes

#### **Optional 6: Loss Landscape Visualization**
**Title:** "Visualizing the Loss Landscape of Neural Nets"
**Authors:** Li, Xu, Taylor, Studer, Goldstein (2018) | **Pages:** 14 | [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)
**Focus:** Loss surface geometry, filter normalization, effect of skip connections
**Why Read:** Visual and geometric understanding of why some networks are easier to optimize than others

#### **Optional 7: Bayesian Hyperparameter Optimization**
**Title:** "Practical Bayesian Optimization of Machine Learning Algorithms"
**Authors:** Snoek, Larochelle, Adams (2012) | **Pages:** 9 | [arXiv:1206.2944](https://arxiv.org/abs/1206.2944)
**Focus:** Gaussian processes for hyperparameter tuning, acquisition functions
**Why Read:** Elegant application of probability (Gaussian processes) to the practical problem of tuning ML models

#### **Optional 8: Adversarial Optimization**
**Title:** "Generative Adversarial Nets"
**Authors:** Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio (2014) | **Pages:** 9 | [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
**Focus:** Minimax optimization, game theory, probability distribution matching
**Why Read:** Foundational paper for generative AI, shows optimization as a two-player game between neural networks

---

### Probability & Statistics (3 papers)

#### **Optional 9: EM Algorithm**
**Title:** "EM Algorithm and Variants: An Informal Tutorial"
**Authors:** Roche (2012) | **Pages:** 20 | [arXiv:1105.1476](https://arxiv.org/abs/1105.1476)
**Focus:** Expectation-Maximization algorithm, mixture models, convergence properties
**Why Read:** Clear tutorial on the EM algorithm, directly relevant to GMM (Application 3)

#### **Optional 10: Information Theory & Deep Learning**
**Title:** "Deep Learning and the Information Bottleneck Principle"
**Authors:** Tishby, Zaslavsky (2015) | **Pages:** 5 | [arXiv:1503.02406](https://arxiv.org/abs/1503.02406)
**Focus:** Information bottleneck, mutual information, deep network layers as compression
**Why Read:** Connects information theory to understanding why deep learning works — layers progressively compress information

#### **Optional 11: Uncertainty in Deep Learning**
**Title:** "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
**Authors:** Gal, Ghahramani (2016) | **Pages:** 12 | [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)
**Focus:** Bayesian inference, model uncertainty, dropout as variational inference
**Why Read:** Beautiful connection between a practical technique (dropout) and Bayesian probability theory

---

### Deep Learning Theory (4 papers)

#### **Optional 12: Neural Network Approximation Theory**
**Title:** "Deep vs. Shallow Networks: An Approximation Theory Perspective"
**Authors:** Mhaskar, Poggio (2016) | **Pages:** 8 | [arXiv:1608.03287](https://arxiv.org/abs/1608.03287)
**Focus:** Function approximation, depth vs width, compositional functions
**Why Read:** Rigorous analysis of why deep networks outperform shallow ones — depth enables exponential efficiency

#### **Optional 13: Sparse Networks**
**Title:** "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
**Authors:** Frankle, Carlin (2019) | **Pages:** ~15 | [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)
**Focus:** Network pruning, sparse subnetworks, weight initialization
**Why Read:** Reveals that small subnetworks within large networks can match full network performance — implications for efficiency

#### **Optional 14: Knowledge Distillation**
**Title:** "Distilling the Knowledge in a Neural Network"
**Authors:** Hinton, Vinyals, Dean (2015) | **Pages:** 9 | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
**Focus:** Model compression, soft targets, temperature scaling of probability distributions
**Why Read:** Shows how probability distributions (softmax outputs) can transfer knowledge between networks

#### **Optional 15: Rethinking Generalization**
**Title:** "Understanding Deep Learning Requires Rethinking Generalization"
**Authors:** Zhang, Bengio, Hardt, Recht, Vinyals (2017) | **Pages:** ~15 | [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
**Focus:** Generalization theory, memorization vs learning, role of regularization
**Why Read:** Challenges classical learning theory by showing neural networks can memorize random labels — a foundational puzzle in ML theory

---

## How to Earn Bonus Points

1. **Read any optional paper(s)** from the list above
2. **Schedule office hours** to discuss the paper with me
3. **Come prepared** with:
   - Summary of main mathematical concepts
   - Connections to course material
   - Questions or insights
   - How it extends your understanding

**Bonus Structure:**

- 1 optional paper discussed = +2% bonus on final grade
- 2 optional papers discussed = +4% bonus on final grade
- 3+ optional papers discussed = +5% bonus on final grade (maximum)

**Quality matters:** Superficial reading won't earn bonus points. Show deep engagement with the mathematical content!

---

## Paper-Topic Coverage Matrix

### Required Papers

| Paper | Topics 1-3 (LA) | Topic 4 (Calc) | Topic 5 (Prob) | Topic 6 (Opt) | Applications |
|-------|----------------|----------------|----------------|---------------|--------------|
| **1. PCA Tutorial** | ✓✓✓ | | | | App 2 |
| **2. Gradient Descent Overview** | | ✓✓ | | ✓✓✓ | Training |
| **3. VAE** | | ✓ | ✓✓✓ | ✓✓ | App 3 |
| **4. Adam Optimizer** | | ✓✓✓ | ✓ | ✓✓ | Training |
| **5. SVM Primer** | ✓✓ | | | ✓✓ | App 4 |

### Optional Papers Coverage

**Linear Algebra Deep Dives:** Optional 1-4 (Transformers, Recommenders, Word Vectors, Normalization)
**Optimization Deep Dives:** Optional 5-8 (Saddle Points, Loss Landscapes, Bayesian Opt, GANs)
**Probability Deep Dives:** Optional 9-11 (EM Algorithm, Information Theory, Bayesian Dropout)
**Deep Learning Theory:** Optional 12-15 (Approximation, Lottery Tickets, Distillation, Generalization)

---

## PhD-Level Papers (Advanced Reading)

**For graduate students and researchers** seeking deeper mathematical foundations. These are comprehensive, longer papers (20-160 pages) that provide rigorous treatments with full proofs and advanced theory.

---

### Foundational Theory (5 papers)

#### **PhD 1: Mathematical Foundations of Deep Learning**
**Title:** "The Modern Mathematics of Deep Learning"
**Authors:** Berner, Grohs, Kutyniok, Petersen (2021) | **Pages:** ~60 | [arXiv:2105.04026](https://arxiv.org/abs/2105.04026)
**Focus:** Complete mathematical treatment of deep learning theory — vector spaces, function approximation, optimization, and generalization
**Why Read:** The most comprehensive mathematical survey connecting linear algebra, analysis, and approximation theory to deep learning

#### **PhD 2: Large-Scale Optimization Theory**
**Title:** "Optimization Methods for Large-Scale Machine Learning"
**Authors:** Bottou, Curtis, Nocedal (2018) | **Pages:** ~90 | [arXiv:1606.04838](https://arxiv.org/abs/1606.04838)
**Focus:** Complete convergence analysis, noise reduction methods, second-order methods, stochastic gradient theory
**Why Read:** The definitive reference on optimization for ML with full mathematical proofs of convergence rates

#### **PhD 3: Variational Inference**
**Title:** "Variational Inference: A Review for Statisticians"
**Authors:** Blei, Kucukelbir, McAuliffe (2017) | **Pages:** ~33 | [arXiv:1601.00670](https://arxiv.org/abs/1601.00670)
**Focus:** Complete variational inference framework, ELBO derivation, mean-field methods, stochastic variational inference
**Why Read:** Rigorous treatment of how optimization approximates Bayesian inference — connects probability theory to practical algorithms

#### **PhD 4: Gradient-Based Training**
**Title:** "Practical Recommendations for Gradient-Based Training of Deep Architectures"
**Authors:** Bengio (2012) | **Pages:** ~35 | [arXiv:1206.5533](https://arxiv.org/abs/1206.5533)
**Focus:** Learning rate theory, momentum analysis, weight initialization, preprocessing, hyperparameter optimization
**Why Read:** Written by Turing Award winner Yoshua Bengio — comprehensive guide to training neural networks with mathematical justification

#### **PhD 5: Kernel Methods and RKHS Theory**
**Title:** "Kernel Methods in Machine Learning"
**Authors:** Hofmann, Schölkopf, Smola (2008) | **Pages:** ~50 | [arXiv:math/0701907](https://arxiv.org/abs/math/0701907)
**Focus:** Reproducing Kernel Hilbert Spaces, Mercer's theorem, kernel design, representer theorem, SVM duality
**Why Read:** Published in Annals of Statistics — the definitive mathematical treatment of kernel methods by the field's pioneers

---

### Advanced Topics (8 papers)

#### **PhD 6: Randomized Numerical Linear Algebra**
**Title:** "Randomized Numerical Linear Algebra: Foundations and Algorithms"
**Authors:** Martinsson, Tropp (2020) | **Pages:** ~90 | [arXiv:2002.01387](https://arxiv.org/abs/2002.01387)
**Focus:** Randomized SVD, matrix sketching, Johnson-Lindenstrauss, low-rank approximation algorithms
**Why Read:** Comprehensive survey of probabilistic approaches to large-scale matrix computations

#### **PhD 7: Random Matrix Theory for Neural Networks**
**Title:** "A Random Matrix Approach to Neural Networks"
**Authors:** Louart, Liao, Couillet (2017) | **Pages:** ~30 | [arXiv:1702.05419](https://arxiv.org/abs/1702.05419)
**Focus:** Spectral analysis of Gram matrices, deterministic equivalents, concentration inequalities
**Why Read:** Applies random matrix theory to rigorously analyze neural network behavior

#### **PhD 8: Convex Optimization**
**Title:** "Convex Optimization: Algorithms and Complexity"
**Authors:** Bubeck (2015) | **Pages:** ~130 | [arXiv:1405.4980](https://arxiv.org/abs/1405.4980)
**Focus:** Black-box optimization, accelerated methods, mirror descent, interior point methods, stochastic methods
**Why Read:** Self-contained monograph covering all of convex optimization theory — the mathematical backbone of ML

#### **PhD 9: Non-Convex Optimization**
**Title:** "Non-convex Optimization for Machine Learning"
**Authors:** Jain, Kar (2017) | **Pages:** ~160 | [arXiv:1712.07897](https://arxiv.org/abs/1712.07897)
**Focus:** Alternating minimization, EM convergence, matrix completion, phase retrieval, non-convex landscape analysis
**Why Read:** Comprehensive treatment of why and when non-convex optimization succeeds in machine learning

#### **PhD 10: Bayesian Optimization**
**Title:** "A Tutorial on Bayesian Optimization"
**Authors:** Frazier (2018) | **Pages:** ~35 | [arXiv:1807.02811](https://arxiv.org/abs/1807.02811)
**Focus:** Gaussian process regression, acquisition functions, theoretical guarantees, practical considerations
**Why Read:** Complete mathematical framework for Bayesian optimization with full derivations

#### **PhD 11: Statistical Learning Theory**
**Title:** "Statistical Learning Theory: Models, Concepts, and Results"
**Authors:** von Luxburg, Schölkopf (2008) | **Pages:** ~26 | [arXiv:0810.4752](https://arxiv.org/abs/0810.4752)
**Focus:** PAC learning, VC dimension, Rademacher complexity, generalization bounds, model selection
**Why Read:** Rigorous overview of the theoretical foundations of learning — when and why algorithms generalize

#### **PhD 12: SVM Theory**
**Title:** "Support Vector Machines with Applications"
**Authors:** Moguerza, Munoz (2006) | **Pages:** ~28 | [arXiv:math/0612817](https://arxiv.org/abs/math/0612817)
**Focus:** SVM duality theory, margin theory, kernel trick proofs, multi-class extensions
**Why Read:** Complete mathematical treatment of SVMs with real-world application examples

#### **PhD 13: Dimensionality Reduction**
**Title:** "A Survey of Dimensionality Reduction Techniques"
**Authors:** Sorzano, Vargas, Pascual Montano (2014) | **Pages:** ~35 | [arXiv:1403.2877](https://arxiv.org/abs/1403.2877)
**Focus:** PCA, MDS, Isomap, LLE, t-SNE, kernel PCA, manifold learning theory
**Why Read:** Comprehensive mathematical survey covering both linear and nonlinear dimensionality reduction methods

---

### Research Frontiers (10 papers)

These papers represent active research frontiers that every PhD student in mathematical ML should understand.

#### **PhD 14: Neural Tangent Kernel**
**Title:** "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
**Authors:** Jacot, Gabriel, Hongler (2018) | [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
**Focus:** Infinite-width neural networks, kernel regime, convergence of gradient descent, lazy training
**Why Read:** Foundational breakthrough showing that infinitely wide neural networks behave as kernel methods — bridges classical kernel theory with deep learning and provides convergence guarantees for gradient descent

#### **PhD 15: Computational Optimal Transport**
**Title:** "Computational Optimal Transport"
**Authors:** Peyré, Cuturi (2019) | [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
**Focus:** Wasserstein distances, Sinkhorn algorithm, entropy regularization, optimal transport theory
**Why Read:** Optimal transport provides the mathematical framework behind Wasserstein GANs, distribution matching, and domain adaptation — essential for modern generative AI and understanding geometry of probability distributions

#### **PhD 16: Matrix Concentration Inequalities**
**Title:** "An Introduction to Matrix Concentration Inequalities"
**Authors:** Tropp (2015) | [arXiv:1501.01571](https://arxiv.org/abs/1501.01571)
**Focus:** Matrix Bernstein, matrix Chernoff, spectral norm bounds, applications to randomized algorithms
**Why Read:** Essential mathematical toolkit for proving guarantees about randomized matrix algorithms, random graphs, and high-dimensional statistics — the probability theory that makes modern ML theory rigorous

#### **PhD 17: Neural Ordinary Differential Equations**
**Title:** "Neural Ordinary Differential Equations"
**Authors:** Chen, Rubanova, Bettencourt, Duvenaud (2018) | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
**Focus:** Continuous-depth networks, adjoint method for backpropagation, ODE solvers as network layers
**Why Read:** NeurIPS Best Paper — reframes neural networks as continuous dynamical systems, connecting differential equations (Topic 4) to deep learning architecture design. Shows how the adjoint method provides memory-efficient backpropagation through ODE solvers

#### **PhD 18: Diffusion Models**
**Title:** "Denoising Diffusion Probabilistic Models"
**Authors:** Ho, Jain, Abbeel (2020) | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
**Focus:** Forward/reverse diffusion processes, variational lower bound, score matching, Markov chains
**Why Read:** The mathematical foundation of modern image generation (DALL-E, Stable Diffusion). Combines probability theory (Markov chains, Gaussian distributions), calculus (score functions, denoising), and optimization in one elegant framework

#### **PhD 19: Normalizing Flows**
**Title:** "Normalizing Flows for Probabilistic Modeling and Inference"
**Authors:** Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan (2021) | [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
**Focus:** Change of variables formula, invertible transformations, Jacobian computation, density estimation
**Why Read:** Comprehensive review connecting the change-of-variables theorem from calculus (Topic 4) and probability distributions (Topic 5) to powerful generative models — demonstrates how mathematical foundations enable practical density estimation

#### **PhD 20: Graph Neural Networks**
**Title:** "Semi-Supervised Classification with Graph Convolutional Networks"
**Authors:** Kipf, Welling (2017) | [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
**Focus:** Spectral graph theory, graph Laplacian, Chebyshev polynomials, message passing
**Why Read:** Bridges spectral graph theory (eigenvalues of the Laplacian) with deep learning — shows how linear algebra on graphs enables learning from relational data (social networks, molecules, knowledge graphs)

#### **PhD 21: Representation Learning**
**Title:** "Representation Learning: A Review and New Perspectives"
**Authors:** Bengio, Courville, Vincent (2014) | [arXiv:1206.5538](https://arxiv.org/abs/1206.5538)
**Focus:** Feature learning, autoencoders, manifold hypothesis, disentangled representations, deep architectures
**Why Read:** Written by Turing Award winner Yoshua Bengio — comprehensive review of how neural networks learn useful representations from data. Connects information theory, manifold geometry, and probabilistic models to understanding what deep networks actually learn

#### **PhD 22: Double Descent and Modern Generalization**
**Title:** "Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-off"
**Authors:** Belkin, Hsu, Ma, Mandal (2019) | [arXiv:1812.11118](https://arxiv.org/abs/1812.11118)
**Focus:** Double descent curve, interpolation threshold, over-parameterization, classical vs modern learning theory
**Why Read:** Challenges the classical bias-variance trade-off by showing that going beyond the interpolation threshold can improve generalization — a fundamental puzzle that reshaped our understanding of why over-parameterized neural networks generalize

#### **PhD 23: Matrix Calculus for Deep Learning**
**Title:** "The Matrix Calculus You Need For Deep Learning"
**Authors:** Parr, Howard (2018) | [arXiv:1802.01528](https://arxiv.org/abs/1802.01528)
**Focus:** Jacobian matrices, chain rule for vectors/matrices, gradient computation, backpropagation derivation
**Why Read:** The most accessible and complete reference for matrix calculus in deep learning — derives all the gradient rules needed for understanding backpropagation, from scalar derivatives to full Jacobian computations. Essential reference for implementing and understanding any neural network

---

## Reading Tips

**Before Reading:**

- Review relevant course topics first
- Have the textbook nearby for reference
- Each paper is ≤20 pages — plan 2-3 hours for focused reading

**During Reading:**

- Work through all mathematical derivations yourself
- Write down questions as they arise
- Make notes connecting to course material
- Don't skip the proofs — they build understanding

**After Reading:**

- Summarize the main mathematical results
- Identify connections to other course topics
- Attempt practice problems if provided
- Prepare your written review

**For Bonus Papers:**

- Choose papers that interest you or align with research goals
- Focus on areas where you want deeper expertise
- Prepare specific questions for office hour discussion

---

<div class="signature" markdown="1">

*Paper selection curated by Mohammed Alnemari*
*Mathematics of AI &bull; Spring 2026*

</div>

---

<div class="last-updated" markdown="1">
**Last Updated:** February 8, 2026
</div>
