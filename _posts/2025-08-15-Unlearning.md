---
title: "Editing Concepts in Diffusion Models"
mathjax: true
layout: post
categories: media
---

![CLIP Editing Results](https://raw.githubusercontent.com/Om2005Prakash/Editing-Concepts-in-Stable-Diffusion/refs/heads/main/assets/results.png)

> **Note:** This is a solution write-up for our submission to the **Unlearning and Model Editing (U&ME) Workshop at ICCV '25**.  
> * [Workshop Website](https://sites.google.com/view/u-and-me-workshop/)
> * [Current Leaderboard](https://shreyanshhub.github.io/GENMU-/leaderboard.html)

*(For the technical solution, skip to the section "Null-Space Constrained Editing")*

## Motivation for Unlearning

Over the past few months, I’ve been exploring a deceptively simple question:
**Can a diffusion model forget — without losing everything else it knows?**

As text-to-image models become increasingly capable, they also inherit the biases, copyrighted material, and harmful content of their massive training datasets. From *LAION-5B* to other web-scale corpora powering today’s generative systems, this raises a critical issue: *how* can we make them unlearn responsibly?

Editing can serve as a form of unlearning. If we can edit an unwanted concept into a safe target concept, we can effectively "forget" the original one. This post describes a method I call **Null-Space Constrained Concept Editing**, adapted from a paper called **AlphaEdit** ([arXiv:2410.02355](https://arxiv.org/abs/2410.02355)), which enables *selective editing* in diffusion models while preserving unrelated knowledge.

In this post, we focus on editing the **CLIP text encoder** to realign the embeddings of unwanted concepts toward safe targets. Crucially, we do **not** modify the UNet weights directly; however, we leverage the UNet to guide the update direction (as detailed in the section *Guidance From the UNet*).



---

## The Problem: Naive Editing

A naive objective for editing a single linear layer \\( W \\) attempts to balance editing new concepts (\\( K_1 \\)) while preserving old ones (\\( K_0 \\)):

$$
\min_{\Delta} \left( 
\underbrace{\| (W + \Delta)K_1 - V_1 \|^2}_{\text{Edit Error}} 
+ 
\underbrace{\| (W + \Delta)K_0 - V_0 \|^2}_{\text{Preservation Error}}
\right)
$$

Here, \\( V_0 = WK_0 \\) represents the original outputs we wish to maintain.

**The issue:** Minimizing the edit error often requires distorting \\( W \\) in directions that inadvertently alter the output for \\( K_0 \\). This leads to the classic problem of **"catastrophic forgetting."**

---

## The Solution: Null-Space Constrained Editing (AlphaEdit)

A recent method, **AlphaEdit**, solves this geometrically. Instead of trying to balance two competing errors, it restricts updates to the **null space** of the preserved knowledge.

The core idea is to construct an update \\( \Delta \\) that acts **only** where \\( K_0 \\) has no presence.



### 1. Constructing the Projector
We treat the update as a low-rank modification projected onto a specific subspace. First, we analyze the covariance of the preservation keys using SVD:

$$
K_0 K_0^T = U \Sigma U^T
$$

We select the eigenvectors in \\( U \\) corresponding to the smallest eigenvalues (effectively zero). Let \\( \tilde{U} \\) be the matrix of these "null" eigenvectors. We define our projection matrix \\( P \\) as:

$$
P = \tilde{U}\tilde{U}^T
$$

**The geometric intuition:** Because \\( \tilde{U} \\) spans the null space of the input correlations, any vector projected by \\( P \\) is orthogonal to the existing knowledge \\( K_0 \\). Therefore:

$$
P K_0 \approx 0
$$

### 2. The Null-Space Objective
We constrain our update to be of the form \\( \Delta P \\). Substituting this into the naive objective:

$$
\min_{\Delta} \left( \| (W + \Delta P)K_1 - V_1 \|^2 + \| (W + \underbrace{\Delta P)K_0}_{\approx 0} - V_0 \|^2 \right)
$$

Because \\( P K_0 \approx 0 \\), the preservation term vanishes naturally (since \\( WK_0 = V_0 \\)). The constraint ensures we **cannot** hurt the old knowledge. The problem simplifies to:

$$
\min_{\Delta} \| (W + \Delta P)K_1 - V_1 \|^2
$$

### 3. Closed-Form Solution
This simplified objective admits a closed-form solution for the update:

$$
\Delta_{\text{edit}} = R K_1^T P (K_1 K_1^T P + \lambda I)^{-1}
$$

Where \\( R = V_1 - W K_1 \\) is the residual error of the unedited model.

By projecting the update through \\( P \\), we ensure edits are applied strictly in the "empty space" of the model's knowledge, allowing us to **forget (errors) without forgetting (facts).**

---

## Making It Work: Guidance From the UNet

A subtle challenge remains:
> How do we choose the target embeddings \\( V_1 \\) for the concepts we want to edit?

A naive choice is \\( V_1 = W K_1^* \\), where \\( K_1^* \\) are embeddings of the target (replacement) concepts. However, this quickly leads to **overfitting** — edits perform well on training prompts but fail on unseen ones.

The fix came from leveraging the **UNet**. I designed a small refinement loop that adjusts \\( V_1 \\) so that the UNet’s outputs for the edited and target prompts align:

$$
V_1^{(t+1)} = V_1^{(t)} - \eta \nabla_{V_1} \, \text{MSE}(\text{UNet}(V_1^{(t)}), \text{UNet}(V_1^*))
$$

This **UNet-guided refinement** provides stable target representations that generalize well, even for indirect prompts that don’t explicitly mention the forgotten concept. For example, after editing "Van Gogh" $\to$ "Monet" in CLIP, the prompt *"starry night"* generated a Monet-style painting, even though "starry night" wasn’t seen during the editing process.

---

## Experiments: Unlearning in the Wild

The approach was evaluated on **Stable Diffusion 1.4**, using 20 diverse concepts (objects, actions, and art styles).

**Experimental Setup:**
* **Dataset:** 20 diverse concepts.
* **Prompts:** 10 prompts per concept (5 direct, 5 indirect).
* **Generation:** 20 images generated per prompt.
* **Evaluation:** **LLaVA v1.5 (7B)** was used to detect whether each image contained the target concept.

**Metrics:**
* **Forget Score:** Fraction of images where the edited concept was *not recognized* by LLaVA.
* **Retain Score:** Fraction of images where unedited concepts were *correctly recognized*.

The results show that our method improves the *forget–retain tradeoff* by roughly **20%** compared to prior baselines, with most concepts unlearned using only **one edit** in CLIP.

| Method | Harmonic Mean (Forget $\times$ Retain) |
| :--- | :---: |
| Unedited Model | 0.313 |
| Unified Concept Editing (UCE) | 0.480 |
| Erasing Stable Diffusion (ESD) | 0.504 |
| **Our Method (Null-Space Editing)** | **0.642** |

---

## Further Reading

If you’d like to dive deeper, the full code and a detailed report of experiments are available on GitHub:

[![GitHub](https://img.shields.io/badge/GitHub-View_Code-black?style=for-the-badge&logo=github)](https://github.com/Om2005Prakash/Editing-Concepts-in-Stable-Diffusion)