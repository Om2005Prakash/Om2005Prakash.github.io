---
title:  "Editing Concepts in Diffusion Models"
mathjax: true
layout: post
categories: media
---

![CLIP Editing Results](https://raw.githubusercontent.com/Om2005Prakash/Editing-Concepts-in-Stable-Diffusion/refs/heads/main/assets/results.png)



Over the past few months, I’ve been exploring a deceptively simple question:  
**Can a diffusion model forget — without losing everything else it knows?**

As text-to-image models become increasingly capable, they also inherit the biases, copyrighted material, and harmful content of their massive training datasets. From datasets like *LAION-5B* to other web-scale corpora powering today’s generative systems, the issue is no longer *if* these models memorize undesirable concepts, but *how* we can make them unlearn responsibly.

Editing can serve as a form of unlearning — if we can edit an unwanted concept into a different, safe target concept, we can effectively “forget” the original one.  
This post describes a method I call **Null-Space Constrained Concept Editing** — an approach that enables *selective forgetting* in diffusion models while preserving unrelated knowledge.

---

## Naive Editing

Knowledge-editing methods have recently gained attention in large language models (LLMs).  
I wanted to test whether those same methods could be applied to the **CLIP text encoder** used in diffusion models, which shares a similar architecture with LLMs.

A naive editing objective for a single linear layer \( W \) can be defined as:

$$
\min_{\tilde{\Delta}} \left( 
\| (W + \tilde{\Delta})K_1 - V_1 \|^2 
+ 
\| (W + \tilde{\Delta})K_0 - V_0 \|^2
\right)
$$

Here:
- \( K_1 \): inputs corresponding to **concepts to be edited**,  
- \( K_0 \): inputs for **concepts to be preserved**,  
- \( V_1 \), \( V_0 \): their respective desired outputs.  

This objective attempts to *edit* certain concepts while *preserving* others. However, in practice, the preserved concepts can still be unintentionally distorted.

---

## Null-Space Constrained Editing

While the naive formulation works in principle, it often interferes with unrelated concepts.  
A recent method called **AlphaEdit** ([arXiv:2410.02355](https://arxiv.org/abs/2410.02355)) brought a major improvement in knowledge editing by introducing a **null-space projection** — ensuring edits don’t overwrite existing knowledge.

Inspired by this, I extended the idea to diffusion models.  
We ensure that updates to \( W \) lie in the *left null space* of \( K_0 \) so that the preservation set remains unaffected.

Formally, if \( U \) is an orthonormal basis for the null space of \( K_0 \) (i.e., \( U^\top K_0 = 0 \)),  
we define the projection matrix:

$$
P = U U^\top
$$

This gives the constrained objective:

$$
\min_{\Delta} \| (W + \Delta P)K_1 - V_1 \|^2
+ 
\| (W + \Delta P)K_0 - V_0 \|^2
$$

Since \( P K_0 = 0 \), the second term vanishes, leaving:

$$
\min_{\Delta} \| (W + \Delta P)K_1 - V_1 \|^2
$$

The closed-form solution for this objective is:

$$
\Delta_{\text{edit}} = R K_1^\top P (K_1 K_1^\top P + I)^{-1}, \quad
R = V_1 - W K_1
$$

This projection ensures that preserved knowledge remains completely intact — allowing the model to *forget without forgetting*.

---

## Making It Work: Guidance From the UNet

A subtle challenge remains:  
> How do we choose the target embeddings \( V_1 \) for the concepts we want to edit?

A naive choice is \( V_1 = W K_1^* \), where \( K_1^* \) are embeddings of the target (replacement) concepts.  
However, this quickly leads to **overfitting** — edits perform well on training prompts but fail on unseen ones.

The fix came from leveraging the **UNet**.  
I designed a small refinement loop that adjusts \( V_1 \) so that the UNet’s outputs for the edited and target prompts align:

$$
V_1^{(t+1)} = V_1^{(t)} - \eta \nabla_{V_1} \, \text{MSE}(\text{UNet}(V_1^{(t)}), \text{UNet}(V_1^*))
$$

This **UNet-guided refinement** provides stable target representations that generalize well — even for indirect prompts that don’t explicitly mention the forgotten concept.  
For example, after editing “Van Gogh” → “Monet” in CLIP, the prompt *“starry night”* generated a Monet-style painting, even though “starry night” wasn’t seen during editing.

---

## Experiments: Unlearning in the Wild

The approach was evaluated on **Stable Diffusion 1.4**, using 20 diverse concepts (objects, actions, and art styles).  
For each concept:
- 10 prompts were used (5 direct, 5 indirect)  
- 20 images were generated per prompt  

**LLaVA v1.5 (7B)** was used to detect whether each image contained the target concept.

Two key metrics were computed:
- **Forget Score** — fraction of images where the edited concept was *not recognized* by LLaVA.  
- **Retain Score** — fraction of images where unedited concepts were *correctly recognized*.

The results show that our method improves the *forget–retain tradeoff* by roughly **20%** compared to prior baselines, with most concepts unlearned using only **one edit** in CLIP.

| Method | Harmonic Mean (Forget ⨉ Retain) |
|:--------|:------------------------------:|
| Unedited Model | 0.313 |
| Unified Concept Editing (UCE) | 0.480 |
| Erasing Stable Diffusion (ESD) | 0.504 |
| **Our Method (Null-Space Editing)** | **0.642** |

![Single Concept Editing](https://raw.githubusercontent.com/Om2005Prakash/Editing-Concepts-in-Stable-Diffusion/refs/heads/main/assets/forget_concept_concept_table.png)

---

## Looking Ahead

The method works remarkably well but raises a few open questions:

- How many concepts can be edited before the model’s coherence breaks down?  
- Can we automate prompt generation to avoid overfitting?  
- Could causal tracing — like in LLMs — reveal *which layers* encode specific concepts in CLIP?

These are the directions I’m currently exploring.  
The long-term goal is to build diffusion models that can **responsibly unlearn** — preserving creativity and safety side by side.

If you’d like to dive deeper, the full code and experiments are available here:  
👉 [Om2005Prakash/Editing-Concepts-in-Stable-Diffusion](https://github.com/Om2005Prakash/Editing-Concepts-in-Stable-Diffusion)

---

*Thanks for reading. If you’re working on diffusion unlearning, I’d love to hear how you’re approaching the “forget without forgetting” challenge.*
