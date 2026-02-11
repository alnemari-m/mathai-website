# Tutorial 5: Probability and Distributions

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

---

## Learning Objectives

By the end of this tutorial, you will understand:

1. Probability spaces, sample spaces, events, and the Kolmogorov axioms
2. Conditional probability and independence
3. Bayes' theorem and how to apply it
4. Discrete random variables and their distributions (Bernoulli, Binomial, Geometric)
5. Continuous random variables and their distributions (Uniform, Exponential, Gaussian)
6. Expected value and variance, including computation rules
7. The Gaussian (Normal) distribution and its properties
8. Joint and marginal distributions
9. Covariance and correlation
10. The sum rule and product rule of probability

---

## Part 1: Probability Space

### 1.1 Core Definitions

A **probability space** is a triple $(\Omega, \mathcal{F}, P)$ consisting of three components:

| Component | Name | Description |
|-----------|------|-------------|
| $\Omega$ | Sample space | The set of all possible outcomes of an experiment |
| $\mathcal{F}$ | Event space | A collection of subsets of $\Omega$ (the events we can assign probabilities to) |
| $P$ | Probability function | A function $P: \mathcal{F} \to [0, 1]$ that assigns probabilities to events |

**Example (Coin Flip):**
- Sample space: $\Omega = \{H, T\}$
- Event space: $\mathcal{F} = \{\emptyset, \{H\}, \{T\}, \{H, T\}\}$
- Probability: $P(\{H\}) = 0.5$, $P(\{T\}) = 0.5$

**Example (Rolling a Die):**
- Sample space: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Event "rolling an even number": $A = \{2, 4, 6\}$
- $P(A) = \frac{3}{6} = \frac{1}{2}$

### 1.2 Kolmogorov Axioms of Probability

All of probability theory rests on three axioms, formalized by Andrey Kolmogorov:

| Axiom | Statement | Meaning |
|-------|-----------|---------|
| **Axiom 1** (Non-negativity) | $P(A) \geq 0$ for every event $A$ | Probabilities are never negative |
| **Axiom 2** (Normalization) | $P(\Omega) = 1$ | Something must happen |
| **Axiom 3** (Additivity) | If $A \cap B = \emptyset$, then $P(A \cup B) = P(A) + P(B)$ | For mutually exclusive events, probabilities add |

**Key consequences of the axioms:**

- $P(\emptyset) = 0$ (the impossible event has probability zero)
- $P(A^c) = 1 - P(A)$ (complement rule)
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (inclusion-exclusion)
- If $A \subseteq B$, then $P(A) \leq P(B)$ (monotonicity)

**Worked Example:**
Suppose $P(A) = 0.6$ and $P(B) = 0.4$ with $P(A \cap B) = 0.2$. Find $P(A \cup B)$.

$$P(A \cup B) = P(A) + P(B) - P(A \cap B) = 0.6 + 0.4 - 0.2 = 0.8$$

---

## Part 2: Conditional Probability and Independence

### 2.1 Conditional Probability

The **conditional probability** of event $A$ given that event $B$ has occurred is:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{provided } P(B) > 0$$

This reads: "the probability of $A$ given $B$."

**Intuition:** Once we know $B$ happened, the sample space effectively shrinks to $B$, and we ask how much of $B$ is also in $A$.

**Worked Example:**
A standard deck of 52 cards. What is the probability a card is a King given it is a face card?

- $B$ = face card: there are 12 face cards (J, Q, K of each suit), so $P(B) = \frac{12}{52}$
- $A \cap B$ = King and face card = King: there are 4 Kings, so $P(A \cap B) = \frac{4}{52}$

$$P(\text{King} \mid \text{Face card}) = \frac{P(A \cap B)}{P(B)} = \frac{4/52}{12/52} = \frac{4}{12} = \frac{1}{3}$$

### 2.2 Independence

Two events $A$ and $B$ are **independent** if knowing one gives no information about the other. Formally:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently, if $A$ and $B$ are independent:

$$P(A \mid B) = P(A) \quad \text{and} \quad P(B \mid A) = P(B)$$

**Example:**
Rolling two fair dice. Let $A$ = "first die shows 3" and $B$ = "second die shows 5."

$$P(A) = \frac{1}{6}, \quad P(B) = \frac{1}{6}, \quad P(A \cap B) = \frac{1}{36} = \frac{1}{6} \cdot \frac{1}{6}$$

Since $P(A \cap B) = P(A)P(B)$, the events are independent.

**Warning:** Independence is not the same as mutual exclusivity. If $A$ and $B$ are mutually exclusive and both have positive probability, they are **not** independent (knowing one happened tells you the other did not).

---

## Part 3: Bayes' Theorem

### 3.1 The Formula

**Bayes' theorem** lets us "reverse" conditional probabilities:

$$\boxed{P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}}$$

| Term | Name | Interpretation |
|------|------|----------------|
| $P(A \mid B)$ | **Posterior** | Updated belief about $A$ after observing $B$ |
| $P(A)$ | **Prior** | Initial belief about $A$ before seeing evidence |
| $P(B \mid A)$ | **Likelihood** | How probable the evidence $B$ is if $A$ is true |
| $P(B)$ | **Evidence** (marginal likelihood) | Total probability of observing $B$ |

### 3.2 The Law of Total Probability

The denominator $P(B)$ is often computed using the **law of total probability**. If $A_1, A_2, \ldots, A_n$ partition $\Omega$:

$$P(B) = \sum_{i=1}^{n} P(B \mid A_i) \, P(A_i)$$

For two complementary events $A$ and $A^c$:

$$P(B) = P(B \mid A) \, P(A) + P(B \mid A^c) \, P(A^c)$$

### 3.3 Worked Example: Medical Testing

A disease affects 1% of the population. A test has:
- **Sensitivity** (true positive rate): $P(\text{Positive} \mid \text{Disease}) = 0.95$
- **Specificity** (true negative rate): $P(\text{Negative} \mid \text{No Disease}) = 0.90$

If a person tests positive, what is $P(\text{Disease} \mid \text{Positive})$?

**Step 1:** Define events and assign values.
- $P(D) = 0.01$, $P(D^c) = 0.99$
- $P(+ \mid D) = 0.95$, $P(+ \mid D^c) = 1 - 0.90 = 0.10$

**Step 2:** Compute $P(+)$ using the law of total probability.

$$P(+) = P(+ \mid D) \, P(D) + P(+ \mid D^c) \, P(D^c)$$
$$P(+) = (0.95)(0.01) + (0.10)(0.99) = 0.0095 + 0.099 = 0.1085$$

**Step 3:** Apply Bayes' theorem.

$$P(D \mid +) = \frac{P(+ \mid D) \, P(D)}{P(+)} = \frac{(0.95)(0.01)}{0.1085} = \frac{0.0095}{0.1085} \approx 0.0876$$

**Interpretation:** Even with a positive test, there is only about an 8.8% chance the person actually has the disease. This counterintuitive result arises because the disease is rare (low prior), so false positives outnumber true positives.

---

## Part 4: Discrete Random Variables

### 4.1 Definitions

A **random variable** $X$ is a function that maps outcomes in the sample space to real numbers:

$$X: \Omega \to \mathbb{R}$$

A random variable is **discrete** if it takes values from a countable set (e.g., $\{0, 1, 2, \ldots\}$).

The **probability mass function (PMF)** of a discrete random variable $X$ is:

$$p(x) = P(X = x)$$

**Properties of a valid PMF:**

1. $p(x) \geq 0$ for all $x$
2. $\displaystyle\sum_{\text{all } x} p(x) = 1$

### 4.2 Bernoulli Distribution

A single trial with two outcomes: success ($X = 1$) with probability $p$, or failure ($X = 0$) with probability $1 - p$.

$$X \sim \text{Bernoulli}(p)$$

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

| Property | Value |
|----------|-------|
| Mean | $E[X] = p$ |
| Variance | $\text{Var}(X) = p(1-p)$ |

**Example:** A coin flip with $P(\text{Heads}) = 0.6$. Then $X \sim \text{Bernoulli}(0.6)$.

### 4.3 Binomial Distribution

The number of successes in $n$ independent Bernoulli trials, each with success probability $p$.

$$X \sim \text{Binomial}(n, p)$$

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, 2, \ldots, n$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient.

| Property | Value |
|----------|-------|
| Mean | $E[X] = np$ |
| Variance | $\text{Var}(X) = np(1-p)$ |

**Worked Example:**
A fair coin is flipped 10 times. What is the probability of getting exactly 4 heads?

$$P(X = 4) = \binom{10}{4} (0.5)^4 (0.5)^{6} = \binom{10}{4} (0.5)^{10}$$
$$= \frac{10!}{4! \cdot 6!} \cdot \frac{1}{1024} = \frac{210}{1024} \approx 0.2051$$

### 4.4 Geometric Distribution

The number of trials until the first success in a sequence of independent Bernoulli trials.

$$X \sim \text{Geometric}(p)$$

$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

| Property | Value |
|----------|-------|
| Mean | $E[X] = \frac{1}{p}$ |
| Variance | $\text{Var}(X) = \frac{1-p}{p^2}$ |

**Worked Example:**
You roll a fair die until you get a 6. What is the probability it takes exactly 3 rolls?

$$P(X = 3) = \left(\frac{5}{6}\right)^{2} \cdot \frac{1}{6} = \frac{25}{36} \cdot \frac{1}{6} = \frac{25}{216} \approx 0.1157$$

The expected number of rolls: $E[X] = \frac{1}{1/6} = 6$.

---

## Part 5: Continuous Random Variables

### 5.1 Definitions

A random variable is **continuous** if it can take any value in an interval (or union of intervals).

The **probability density function (PDF)** $f(x)$ satisfies:

1. $f(x) \geq 0$ for all $x$
2. $\displaystyle\int_{-\infty}^{\infty} f(x) \, dx = 1$
3. $\displaystyle P(a \leq X \leq b) = \int_{a}^{b} f(x) \, dx$

**Important:** For a continuous random variable, $P(X = x) = 0$ for any specific value $x$. Only intervals have nonzero probability.

The **cumulative distribution function (CDF)** is:

$$F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) \, dt$$

**Properties of the CDF:**
- $F(-\infty) = 0$ and $F(\infty) = 1$
- $F$ is non-decreasing
- $f(x) = \frac{d}{dx} F(x)$ (the PDF is the derivative of the CDF)

### 5.2 Uniform Distribution

A random variable is equally likely to take any value in the interval $[a, b]$.

$$X \sim \text{Uniform}(a, b)$$

$$f(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}$$

| Property | Value |
|----------|-------|
| Mean | $E[X] = \frac{a + b}{2}$ |
| Variance | $\text{Var}(X) = \frac{(b - a)^2}{12}$ |

**Example:** If $X \sim \text{Uniform}(0, 10)$, then $E[X] = 5$ and $P(2 \leq X \leq 5) = \frac{5-2}{10-0} = 0.3$.

### 5.3 Exponential Distribution

Models the time between events in a Poisson process. The parameter $\lambda > 0$ is the rate.

$$X \sim \text{Exponential}(\lambda)$$

$$f(x) = \begin{cases} \lambda e^{-\lambda x} & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$$

$$F(x) = 1 - e^{-\lambda x}, \quad x \geq 0$$

| Property | Value |
|----------|-------|
| Mean | $E[X] = \frac{1}{\lambda}$ |
| Variance | $\text{Var}(X) = \frac{1}{\lambda^2}$ |

**Key property (Memoryless):**

$$P(X > s + t \mid X > s) = P(X > t)$$

**Worked Example:**
Light bulbs fail at a rate of $\lambda = 0.01$ per hour. What is the probability a bulb lasts more than 200 hours?

$$P(X > 200) = 1 - F(200) = e^{-0.01 \cdot 200} = e^{-2} \approx 0.1353$$

### 5.4 Gaussian (Normal) Distribution

The most important distribution in statistics and machine learning.

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

| Property | Value |
|----------|-------|
| Mean | $E[X] = \mu$ |
| Variance | $\text{Var}(X) = \sigma^2$ |

Full details on the Gaussian are in Part 7 below.

---

## Part 6: Expected Value and Variance

### 6.1 Expected Value (Mean)

The **expected value** is the long-run average of a random variable.

**Discrete case:**

$$E[X] = \sum_{x} x \, p(x)$$

**Continuous case:**

$$E[X] = \int_{-\infty}^{\infty} x \, f(x) \, dx$$

**Expected value of a function $g(X)$:**

$$E[g(X)] = \sum_{x} g(x) \, p(x) \quad \text{(discrete)}$$

$$E[g(X)] = \int_{-\infty}^{\infty} g(x) \, f(x) \, dx \quad \text{(continuous)}$$

### 6.2 Properties of Expected Value

| Property | Formula |
|----------|---------|
| Linearity | $E[aX + b] = aE[X] + b$ |
| Sum | $E[X + Y] = E[X] + E[Y]$ (always, even if dependent) |
| Product (independent) | $E[XY] = E[X] \cdot E[Y]$ (only if $X, Y$ are independent) |
| Constant | $E[c] = c$ |

**Worked Example:**
Let $X$ be a die roll. Compute $E[X]$.

$$E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = \frac{21}{6} = 3.5$$

### 6.3 Variance

**Variance** measures how spread out a distribution is around its mean.

$$\text{Var}(X) = E\left[(X - E[X])^2\right]$$

**Shortcut formula (very useful for computation):**

$$\boxed{\text{Var}(X) = E[X^2] - (E[X])^2}$$

The **standard deviation** is $\sigma = \sqrt{\text{Var}(X)}$.

### 6.4 Properties of Variance

| Property | Formula |
|----------|---------|
| Scaling | $\text{Var}(aX) = a^2 \text{Var}(X)$ |
| Shift | $\text{Var}(X + b) = \text{Var}(X)$ |
| Affine | $\text{Var}(aX + b) = a^2 \text{Var}(X)$ |
| Sum (independent) | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (only if independent) |
| Constant | $\text{Var}(c) = 0$ |

**Worked Example:**
Let $X$ be a die roll. Compute $\text{Var}(X)$.

First, compute $E[X^2]$:

$$E[X^2] = \frac{1}{6}(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) = \frac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6} \approx 15.167$$

We already know $E[X] = 3.5$, so $(E[X])^2 = 12.25$.

$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - \frac{49}{4} = \frac{182}{12} - \frac{147}{12} = \frac{35}{12} \approx 2.917$$

---

## Part 7: Gaussian (Normal) Distribution in Depth

### 7.1 Definition

The Gaussian distribution with mean $\mu$ and variance $\sigma^2$ has PDF:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}$$

We write $X \sim \mathcal{N}(\mu, \sigma^2)$.

### 7.2 The Standard Normal Distribution

When $\mu = 0$ and $\sigma^2 = 1$, we get the **standard normal** $Z \sim \mathcal{N}(0, 1)$:

$$\phi(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)$$

**Standardization:** Any normal random variable can be converted to a standard normal:

$$Z = \frac{X - \mu}{\sigma}$$

### 7.3 Key Properties

| Property | Description |
|----------|-------------|
| Symmetry | The PDF is symmetric about $\mu$ |
| 68-95-99.7 Rule | ~68% of values fall within $\mu \pm \sigma$, ~95% within $\mu \pm 2\sigma$, ~99.7% within $\mu \pm 3\sigma$ |
| Linear closure | If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $aX + b \sim \mathcal{N}(a\mu + b, \, a^2\sigma^2)$ |
| Sum of normals | If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent, then $X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \, \sigma_1^2 + \sigma_2^2)$ |

**Worked Example:**
Exam scores are distributed as $X \sim \mathcal{N}(75, 100)$ (mean 75, standard deviation 10). What fraction of students score above 90?

Standardize:

$$Z = \frac{90 - 75}{10} = 1.5$$

$$P(X > 90) = P(Z > 1.5) = 1 - \Phi(1.5) \approx 1 - 0.9332 = 0.0668$$

About 6.7% of students score above 90.

### 7.4 Why the Gaussian Matters in Machine Learning

- The **Central Limit Theorem** states that the sum of many independent random variables tends toward a Gaussian, regardless of their individual distributions.
- Many ML algorithms assume Gaussian noise (linear regression, Gaussian processes).
- The multivariate Gaussian is fundamental to dimensionality reduction (PCA) and generative models.

---

## Part 8: Joint and Marginal Distributions

### 8.1 Joint Distribution

The **joint distribution** describes the probability behavior of two (or more) random variables simultaneously.

**Discrete (Joint PMF):**

$$p(x, y) = P(X = x, Y = y)$$

Properties:
- $p(x, y) \geq 0$ for all $x, y$
- $\displaystyle\sum_{x}\sum_{y} p(x, y) = 1$

**Continuous (Joint PDF):**

$$f(x, y) \geq 0 \quad \text{and} \quad \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(x, y) \, dx \, dy = 1$$

### 8.2 Marginal Distribution

The **marginal distribution** of one variable is obtained by summing (or integrating) over the other variable.

**Discrete:**

$$p_X(x) = \sum_{y} p(x, y) \qquad \text{and} \qquad p_Y(y) = \sum_{x} p(x, y)$$

**Continuous:**

$$f_X(x) = \int_{-\infty}^{\infty} f(x, y) \, dy \qquad \text{and} \qquad f_Y(y) = \int_{-\infty}^{\infty} f(x, y) \, dx$$

**Worked Example (Discrete):**
Consider two discrete random variables $X$ and $Y$ with the following joint PMF table:

| | $Y = 0$ | $Y = 1$ | $p_X(x)$ |
|---|---------|---------|-----------|
| $X = 0$ | 0.1 | 0.2 | **0.3** |
| $X = 1$ | 0.3 | 0.4 | **0.7** |
| $p_Y(y)$ | **0.4** | **0.6** | **1.0** |

Marginals are computed by summing each row or column:
- $p_X(0) = 0.1 + 0.2 = 0.3$
- $p_X(1) = 0.3 + 0.4 = 0.7$
- $p_Y(0) = 0.1 + 0.3 = 0.4$
- $p_Y(1) = 0.2 + 0.4 = 0.6$

### 8.3 Independence of Random Variables

$X$ and $Y$ are **independent** if and only if:

$$p(x, y) = p_X(x) \cdot p_Y(y) \quad \text{for all } x, y$$

Check the example above: $p(0, 0) = 0.1$ but $p_X(0) \cdot p_Y(0) = 0.3 \times 0.4 = 0.12 \neq 0.1$. So $X$ and $Y$ are **not** independent.

---

## Part 9: Covariance and Correlation

### 9.1 Covariance

**Covariance** measures how two random variables vary together:

$$\text{Cov}(X, Y) = E\left[(X - E[X])(Y - E[Y])\right]$$

**Shortcut formula:**

$$\boxed{\text{Cov}(X, Y) = E[XY] - E[X] \cdot E[Y]}$$

| Value | Interpretation |
|-------|---------------|
| $\text{Cov}(X,Y) > 0$ | $X$ and $Y$ tend to increase together |
| $\text{Cov}(X,Y) < 0$ | When $X$ increases, $Y$ tends to decrease |
| $\text{Cov}(X,Y) = 0$ | No linear relationship (uncorrelated) |

**Properties of Covariance:**

- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ (symmetric)
- $\text{Cov}(aX + b, \, cY + d) = ac \, \text{Cov}(X, Y)$
- If $X$ and $Y$ are independent, then $\text{Cov}(X, Y) = 0$ (the converse is not always true)

### 9.2 Correlation Coefficient

The **Pearson correlation coefficient** normalizes covariance to the range $[-1, 1]$:

$$\boxed{\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)} \cdot \sqrt{\text{Var}(Y)}} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}}$$

| Value | Interpretation |
|-------|---------------|
| $\rho = 1$ | Perfect positive linear relationship |
| $\rho = -1$ | Perfect negative linear relationship |
| $\rho = 0$ | No linear relationship (uncorrelated) |
| $0 < \rho < 1$ | Positive linear tendency |
| $-1 < \rho < 0$ | Negative linear tendency |

### 9.3 Variance of a Sum (General Case)

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X, Y)$$

If $X$ and $Y$ are independent (so $\text{Cov}(X,Y) = 0$):

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

**Worked Example:**
Using the joint PMF from Part 8, compute $\text{Cov}(X, Y)$.

From the table: $E[X] = 0(0.3) + 1(0.7) = 0.7$ and $E[Y] = 0(0.4) + 1(0.6) = 0.6$.

$$E[XY] = \sum_x \sum_y xy \, p(x,y) = (0)(0)(0.1) + (0)(1)(0.2) + (1)(0)(0.3) + (1)(1)(0.4) = 0.4$$

$$\text{Cov}(X,Y) = E[XY] - E[X]E[Y] = 0.4 - (0.7)(0.6) = 0.4 - 0.42 = -0.02$$

The small negative covariance indicates a very slight negative association.

---

## Part 10: Sum Rule and Product Rule

### 10.1 The Two Fundamental Rules

These two rules form the foundation of all probabilistic reasoning.

**Product Rule (Chain Rule):**

$$P(A, B) = P(A \mid B) \, P(B) = P(B \mid A) \, P(A)$$

This generalizes to multiple variables:

$$P(A, B, C) = P(A \mid B, C) \, P(B \mid C) \, P(C)$$

**Sum Rule (Marginalization):**

$$P(A) = \sum_{B} P(A, B) = \sum_{B} P(A \mid B) \, P(B)$$

### 10.2 Discrete Case

For discrete random variables $X$ and $Y$:

**Product rule:**

$$p(x, y) = p(x \mid y) \, p(y) = p(y \mid x) \, p(x)$$

**Sum rule (marginalization):**

$$p(x) = \sum_{y} p(x, y) = \sum_{y} p(x \mid y) \, p(y)$$

### 10.3 Continuous Case

For continuous random variables $X$ and $Y$:

**Product rule:**

$$f(x, y) = f(x \mid y) \, f(y) = f(y \mid x) \, f(x)$$

**Sum rule (marginalization):**

$$f(x) = \int_{-\infty}^{\infty} f(x, y) \, dy = \int_{-\infty}^{\infty} f(x \mid y) \, f(y) \, dy$$

### 10.4 Connection to Bayes' Theorem

Bayes' theorem is a direct consequence of applying the product rule in both directions and then dividing:

$$f(y \mid x) = \frac{f(x \mid y) \, f(y)}{f(x)} = \frac{f(x \mid y) \, f(y)}{\int f(x \mid y') \, f(y') \, dy'}$$

The denominator uses the sum rule to compute the marginal $f(x)$.

**Worked Example:**
Suppose $Y \in \{0, 1\}$ with $P(Y=1) = 0.3$ and $P(Y=0) = 0.7$. Also:
- $P(X = 1 \mid Y = 1) = 0.9$
- $P(X = 1 \mid Y = 0) = 0.2$

Find $P(Y = 1 \mid X = 1)$ using the sum and product rules.

**Step 1 (Sum rule):** Compute $P(X = 1)$.

$$P(X=1) = P(X=1 \mid Y=1)P(Y=1) + P(X=1 \mid Y=0)P(Y=0)$$
$$= (0.9)(0.3) + (0.2)(0.7) = 0.27 + 0.14 = 0.41$$

**Step 2 (Bayes via product rule):**

$$P(Y=1 \mid X=1) = \frac{P(X=1 \mid Y=1) P(Y=1)}{P(X=1)} = \frac{(0.9)(0.3)}{0.41} = \frac{0.27}{0.41} \approx 0.6585$$

---

## Reference: Table of Common Distributions

### Discrete Distributions

| Distribution | PMF $P(X = k)$ | Mean $E[X]$ | Variance $\text{Var}(X)$ |
|-------------|-----------------|-------------|--------------------------|
| $\text{Bernoulli}(p)$ | $p^k(1-p)^{1-k}$, $k \in \{0,1\}$ | $p$ | $p(1-p)$ |
| $\text{Binomial}(n, p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$, $k = 0,\ldots,n$ | $np$ | $np(1-p)$ |
| $\text{Geometric}(p)$ | $(1-p)^{k-1}p$, $k = 1, 2, \ldots$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ |
| $\text{Poisson}(\lambda)$ | $\frac{\lambda^k e^{-\lambda}}{k!}$, $k = 0, 1, 2, \ldots$ | $\lambda$ | $\lambda$ |

### Continuous Distributions

| Distribution | PDF $f(x)$ | Mean $E[X]$ | Variance $\text{Var}(X)$ |
|-------------|------------|-------------|--------------------------|
| $\text{Uniform}(a,b)$ | $\frac{1}{b-a}$ for $x \in [a,b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| $\text{Exponential}(\lambda)$ | $\lambda e^{-\lambda x}$ for $x \geq 0$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ |
| $\text{Beta}(\alpha, \beta)$ | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ for $x \in [0,1]$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

---

## Summary: Key Takeaways

### Probability Foundations
- A probability space is $(\Omega, \mathcal{F}, P)$ satisfying the Kolmogorov axioms
- Conditional probability: $P(A \mid B) = P(A \cap B) / P(B)$
- Bayes' theorem: $P(A \mid B) = P(B \mid A) P(A) / P(B)$

### Random Variables
- Discrete: described by PMFs; Continuous: described by PDFs
- CDF: $F(x) = P(X \leq x)$ works for both types

### Key Formulas
- Expected value: $E[X] = \sum x \, p(x)$ or $\int x \, f(x) \, dx$
- Variance shortcut: $\text{Var}(X) = E[X^2] - (E[X])^2$
- Covariance: $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$
- Correlation: $\rho_{XY} = \text{Cov}(X,Y) / (\sigma_X \sigma_Y)$

### Fundamental Rules
- Product rule: $P(A, B) = P(A \mid B) P(B)$
- Sum rule: $P(A) = \sum_B P(A, B)$

---

## Practice Problems

### Problem 1
A bag contains 5 red balls and 3 blue balls. Two balls are drawn without replacement. What is the probability that both are red?

### Problem 2
A factory has two machines. Machine A produces 60% of items and Machine B produces 40%. The defect rate is 3% for Machine A and 5% for Machine B. If an item is found to be defective, what is the probability it came from Machine A?

### Problem 3
Let $X \sim \text{Binomial}(8, 0.3)$. Compute $P(X = 2)$, $E[X]$, and $\text{Var}(X)$.

### Problem 4
Let $X \sim \mathcal{N}(50, 25)$ (mean 50, variance 25, so $\sigma = 5$). Find:
- (a) $P(X > 60)$
- (b) $P(40 < X < 55)$

### Problem 5
Random variables $X$ and $Y$ have $E[X] = 3$, $E[Y] = 5$, $E[X^2] = 13$, $E[Y^2] = 30$, and $E[XY] = 16$. Compute $\text{Cov}(X,Y)$, $\text{Var}(X)$, $\text{Var}(Y)$, and the correlation $\rho_{XY}$.

### Problem 6
Consider the continuous random variable $X$ with PDF:

$$f(x) = \begin{cases} cx^2 & \text{if } 0 \leq x \leq 2 \\ 0 & \text{otherwise} \end{cases}$$

- (a) Find the constant $c$ so that $f$ is a valid PDF.
- (b) Compute $E[X]$.
- (c) Compute $\text{Var}(X)$.

---

## Solutions

**Solution 1:**

Use the product rule (chain rule) for drawing without replacement.

$$P(\text{both red}) = P(R_1) \cdot P(R_2 \mid R_1) = \frac{5}{8} \cdot \frac{4}{7} = \frac{20}{56} = \frac{5}{14} \approx 0.3571$$

---

**Solution 2:**

Apply Bayes' theorem. Let $A$ = "from Machine A," $B$ = "from Machine B," and $D$ = "defective."

- $P(A) = 0.6$, $P(B) = 0.4$
- $P(D \mid A) = 0.03$, $P(D \mid B) = 0.05$

First, compute $P(D)$ using the law of total probability:

$$P(D) = P(D \mid A)P(A) + P(D \mid B)P(B) = (0.03)(0.6) + (0.05)(0.4) = 0.018 + 0.02 = 0.038$$

Then:

$$P(A \mid D) = \frac{P(D \mid A) P(A)}{P(D)} = \frac{(0.03)(0.6)}{0.038} = \frac{0.018}{0.038} \approx 0.4737$$

There is about a 47.4% chance the defective item came from Machine A.

---

**Solution 3:**

$X \sim \text{Binomial}(8, 0.3)$.

$$P(X = 2) = \binom{8}{2}(0.3)^2(0.7)^6 = 28 \cdot 0.09 \cdot 0.117649 = 28 \cdot 0.01058841 \approx 0.2965$$

$$E[X] = np = 8 \times 0.3 = 2.4$$

$$\text{Var}(X) = np(1-p) = 8 \times 0.3 \times 0.7 = 1.68$$

---

**Solution 4:**

$X \sim \mathcal{N}(50, 25)$, so $\mu = 50$ and $\sigma = 5$.

**(a)** Standardize:

$$Z = \frac{60 - 50}{5} = 2.0$$

$$P(X > 60) = P(Z > 2.0) = 1 - \Phi(2.0) \approx 1 - 0.9772 = 0.0228$$

About 2.3% of the distribution lies above 60.

**(b)** Standardize both bounds:

$$Z_1 = \frac{40 - 50}{5} = -2.0, \quad Z_2 = \frac{55 - 50}{5} = 1.0$$

$$P(40 < X < 55) = \Phi(1.0) - \Phi(-2.0) \approx 0.8413 - 0.0228 = 0.8185$$

About 81.9% of the distribution falls between 40 and 55.

---

**Solution 5:**

**Covariance:**

$$\text{Cov}(X,Y) = E[XY] - E[X]E[Y] = 16 - (3)(5) = 16 - 15 = 1$$

**Variance of $X$:**

$$\text{Var}(X) = E[X^2] - (E[X])^2 = 13 - 9 = 4 \quad \Rightarrow \quad \sigma_X = 2$$

**Variance of $Y$:**

$$\text{Var}(Y) = E[Y^2] - (E[Y])^2 = 30 - 25 = 5 \quad \Rightarrow \quad \sigma_Y = \sqrt{5}$$

**Correlation:**

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{1}{2\sqrt{5}} = \frac{1}{2\sqrt{5}} \cdot \frac{\sqrt{5}}{\sqrt{5}} = \frac{\sqrt{5}}{10} \approx 0.2236$$

A moderate positive linear association.

---

**Solution 6:**

**(a)** For $f$ to be a valid PDF, the total area must equal 1:

$$\int_0^2 cx^2 \, dx = c \left[\frac{x^3}{3}\right]_0^2 = c \cdot \frac{8}{3} = 1$$

$$c = \frac{3}{8}$$

**(b)** Expected value:

$$E[X] = \int_0^2 x \cdot \frac{3}{8}x^2 \, dx = \frac{3}{8}\int_0^2 x^3 \, dx = \frac{3}{8}\left[\frac{x^4}{4}\right]_0^2 = \frac{3}{8} \cdot \frac{16}{4} = \frac{3}{8} \cdot 4 = \frac{3}{2} = 1.5$$

**(c)** First compute $E[X^2]$:

$$E[X^2] = \int_0^2 x^2 \cdot \frac{3}{8}x^2 \, dx = \frac{3}{8}\int_0^2 x^4 \, dx = \frac{3}{8}\left[\frac{x^5}{5}\right]_0^2 = \frac{3}{8} \cdot \frac{32}{5} = \frac{96}{40} = \frac{12}{5} = 2.4$$

Then apply the variance shortcut:

$$\text{Var}(X) = E[X^2] - (E[X])^2 = 2.4 - (1.5)^2 = 2.4 - 2.25 = 0.15$$

Equivalently, $\text{Var}(X) = \frac{3}{20}$.

---

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

**Previous:** Tutorial 4 - Matrix Decompositions
**Next:** Tutorial 6 - Optimization and Gradient Descent
