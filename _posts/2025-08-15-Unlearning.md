---
title:  "Unlearning Without Forgetting: A Journey to Safer Diffusion Models"
mathjax: true
layout: post
categories: media
---

Can a diffusion model forget — without losing everything else it knows?


Over the past few months, I’ve been exploring a deceptively simple question:  
**Can a diffusion model forget — without losing everything else it knows?**  

As text-to-image models become increasingly capable, they also inherit the biases, copyrighted material, and harmful content of their massive training datasets. From the *LAION-5B* dataset to the web-scale corpora powering today’s generative models, the issue is no longer *if* these systems memorize undesirable concepts, but *how* we can make them unlearn responsibly.

This post shares my journey developing a method I call **Null-Space Constrained Concept Editing** — an approach that enables *selective forgetting* in diffusion models while preserving unrelated knowledge.  
It’s a story of tinkering, equations, and the occasional moment of “wait, that actually works?”

---

## Why We Need to Unlearn

Diffusion models such as Stable Diffusion 1.4 have revolutionized creative generation. But beneath their breathtaking outputs lies a problem: these models often reproduce *sensitive*, *harmful*, or *copyrighted* concepts that were present in the training data.  

Legal frameworks like **GDPR** and **CCPA** already demand a “right to be forgotten.” Yet in practice, retraining massive models from scratch every time someone or something needs to be “forgotten” is infeasible.  

So, the question becomes:  
> How can we surgically remove certain knowledge from a diffusion model *without retraining it* — and without damaging everything else?

That question guided the rest of this work.

---

## The Idea: Editing the Text Encoder, Not the Image Generator

Most concept-editing techniques target the **U-Net** — the heart of the diffusion model that turns noise into images. But what if the real key lies elsewhere?  

I realized that **CLIP’s text encoder** — the part that converts text prompts into embeddings — holds the conceptual DNA of the model. If we can modify how the text encoder represents certain ideas, we can effectively “forget” them at the root, before they even influence the image synthesis.

So, I formulated a new editing objective for a single linear layer $$ W $$ in CLIP:

$$
\min_{\tilde{\Delta}} \left( 
\| (W + \tilde{\Delta})K_1 - V_1 \|^2 
+ 
\| (W + \tilde{\Delta})K_0 - V_0 \|^2
\right)
$$

Here:
- $$ K_1 $$ are the inputs corresponding to **concepts to be edited**,  
- $$ K_0 $$ are the inputs for **concepts to be preserved**,  
- $$ V_1 $$ and $$ V_0 $$ are their desired outputs.  

This objective tries to *edit* certain concepts while *preserving* others.

---

## The Key Insight: The Null Space Trick

While the above works in principle, it can unintentionally distort preserved concepts.  
The fix came from a beautiful piece of linear algebra: the **null space**.

If we ensure that updates to $$ W $$ lie in the *left null space* of $$ K_0 $$, then the change won’t affect the preservation set at all.

Formally, if $$ U $$ is an orthonormal basis for the null space of $$ K_0 $$ (so that $$ U^\top K_0 = 0 $$),  
we define a projection matrix:

$$
P = U U^\top
$$

This gives a new constrained objective:

$$
\min_{\Delta} \| (W + \Delta P)K_1 - V_1 \|^2
$$

And its closed-form update turns out to be:

$$
\Delta_{\text{edit}} = R K_1^\top P (K_1 K_1^\top P + I)^{-1}
$$
where $$ R = V_1 - W K_1 $$.

This simple projection guarantees that preserved knowledge remains intact — the model truly *forgets without forgetting*.

---

## Making It Work: Guidance From the UNet

A practical question arose:  
> How do we choose the target embeddings $$ V_1 $$ for the concepts we want to edit?

At first, I simply set $$ V_1 = W K_1^* $$, where $$ K_1^* $$ are the embeddings of the target (replacement) concepts.  
But that quickly led to **overfitting** — the edits worked on direct prompts but failed on indirect ones.

The fix came from the **UNet itself**.  
I designed a small refinement loop where $$ V_1 $$ is adjusted to ensure the UNet’s outputs for the edited and target prompts align:

$$
V_1^{(t+1)} = V_1^{(t)} - \eta \nabla_{V_1} \, \text{MSE}(\text{UNet}(V_1^{(t)}), \text{UNet}(V_1^*))
$$

This **UNet-guided refinement** provided stable targets that generalize well — even for indirect prompts that don’t mention the forgotten concept explicitly.

---

## Experiments: Unlearning in the Wild

I evaluated this approach on **Stable Diffusion 1.4**, focusing on 20 diverse concepts (objects, art styles, and actions).  
For each concept, I generated:
- 10 prompts (5 direct, 5 indirect)
- 20 images per prompt  
and used **LLaVA v1.5 (7B)** to check whether the concept still appeared.

Two key metrics guided the evaluation:
- **Forget Score** — how effectively the concept was erased.
- **Retain Score** — how well unrelated concepts were preserved.

Here’s the punchline:  
My method achieved a **20% higher harmonic mean** of forget and retain scores than previous baselines like ESD and UCE.  
Even more excitingly, most concepts could be unlearned with *just one edit* in CLIP — compared to 30–40 edits when modifying the UNet directly.

| Method | Harmonic Mean (Forget ⨉ Retain) |
|:--------|:------------------------------:|
| Unedited Model | 0.313 |
| Unified Concept Editing (UCE) | 0.480 |
| Erasing Stable Diffusion (ESD) | 0.504 |
| **Our Method (Null-Space Editing)** | **0.642** |

---

## What I Learned

A few lessons stood out:

1. **CLIP is where concepts live.**  
   Editing CLIP’s feed-forward layers is vastly more effective than touching the UNet.

2. **Linear algebra is your friend.**  
   The null-space constraint might sound abstract, but it elegantly enforces “do no harm” to preserved knowledge.

3. **Small edits can go a long way.**  
   A single-layer edit in CLIP can shift entire conceptual behaviors in the generated images.

4. **Evaluation matters.**  
   Using a vision-language model like LLaVA for semantic checking turned out to be a robust way to measure forgetting.

---

## Looking Ahead

This method works surprisingly well, but it raises deeper questions:

- How many concepts can we edit before the model’s coherence breaks down?  
- Can we automate prompt generation to prevent overfitting?  
- Could causal tracing — similar to language model editing — reveal *which layers* encode specific concepts in CLIP?

These are the threads I’m currently exploring.  
The dream is a future where we can **responsibly unlearn** — where generative models respect privacy, creativity, and control without sacrificing their brilliance.

If you’d like to dive deeper, the full code and experiments are available on GitHub:  
👉 [Om2005Prakash/Editing-Concepts-in-Stable-Diffusion](https://github.com/Om2005Prakash/Editing-Concepts-in-Stable-Diffusion)

---

*Thanks for reading. If you’re working on diffusion unlearning, I’d love to hear how you’re approaching the “forget without forgetting” challenge.*
