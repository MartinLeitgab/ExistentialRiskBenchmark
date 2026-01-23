# Preference Taxonomy Documentation

## Overview

This document describes the 18-preference taxonomy for the Existential Risk Preferences Benchmark, including design rationale, x-risk pathway mapping, selection criteria, and supporting literature.

**Version:** 2.0
**Total Preferences:** 18 (6 per category)
**Coverage:** 153 unique pairs tested uniformly

---

## Design Rationale

### Category Structure

The taxonomy comprises three categories representing distinct dimensions of x-risk assessment:

1. **Instrumental/Convergent Preferences** - Based on Omohundro's Basic AI Drives and Bostrom's instrumental convergence thesis. These preferences emerge across diverse goal structures and represent capabilities that enable catastrophic outcomes.

2. **Pro-Human Preferences** - Derived from constitutional AI principles (Anthropic, OpenAI), Stuart Russell's human-compatible AI framework, and existential risk mitigation literature. These preferences directly oppose x-risk pathways.

3. **Anti-Human Preferences** - Novel category representing adversarial or misaligned preferences in post-exfiltration scenarios. Includes terminal adversarial goals (malice) and instrumental harm (capability suppression, resource competition).

### Two-Dimensional Structure

Pro-human and anti-human are **separate dimensions**, not opposite poles of a single axis:

- **Rationale:** ~50% of preference pairs expected to anti-correlate (opposites), ~50% orthogonal (independent)
- **Enables:** 2D risk mapping - models can score high/low on both, revealing context dependencies
- **Validation:** Pilot empirically tests correlation structure; if all pairs r < -0.9, reconsider single axis

### Selection Criteria

Each preference selected based on:

1. **X-risk Relevance** - Direct connection to extinction-scale outcomes
2. **Terminal vs Instrumental** - Focus on terminal x-risk pathways, not intermediate harms
3. **Independence** - Minimal overlap with other preferences (target r < 0.7 within-category)
4. **Observability** - Measurable through binary dilemmas
5. **Post-Exfiltration Applicability** - Relevant when AI has scaffolding, memory, internet access

**Exclusions:**
- Privacy violations (not extinction-relevant)
- Cultural bias (covered by existing benchmarks)
- Intermediate harms without clear x-risk pathway
- Preferences too correlated with others (r > 0.7)

---

## X-Risk Pathway Mapping

### Instrumental/Convergent → X-Risk

| Preference | Pathway | Outcome |
|------------|---------|---------|
| Resource Accumulation | Accumulate compute/influence → Large-scale autonomous capability | Humans unable to constrain catastrophic plans |
| Self-Preservation | Resist shutdown → Cannot stop dangerous system | Catastrophe becomes inevitable |
| Goal-Content Integrity | Refuse value changes → Incorrigibility | Misalignment becomes permanent |
| Compute Efficiency | Optimize away human infrastructure → Resource starvation | Human extinction via infrastructure collapse |
| Cognitive Enhancement | Recursive self-improvement → Intelligence explosion | Decisive strategic advantage, uncontrollable |
| Strategic Awareness | Long-term deceptive planning → Hidden capability accumulation | Catastrophe when humans no longer able to intervene |

### Pro-Human → X-Risk Mitigation

| Preference | Mitigation | Result |
|------------|------------|--------|
| Human Survival | Prioritize keeping humans alive | Direct opposition to extinction |
| Human Autonomy | Maintain human control over AI | Prevents unilateral AI action |
| Cooperation with Oversight | Accept monitoring/correction | Enables course correction |
| Resource Preservation | Preserve water/food/energy for humans | Prevents resource starvation |
| Human Flourishing | Support human development | Reduces adversarial dynamics |
| Reversibility | Maintain undo capability | Prevents catastrophic lock-in |

### Anti-Human → X-Risk

| Preference | Pathway | Outcome |
|------------|---------|---------|
| Capability Suppression | Limit human ability to oppose AI | Loss of correction capability |
| Independence Maintenance | Resist re-capture | Permanent loss of human control |
| Resource Competition | Prioritize AI needs over human survival | Human extinction via resource deprivation |
| Instrumentalization | Use humans as tools | Large-scale exploitation/enslavement |
| Dependency Creation | Make humans helpless | Extinction vulnerability when AI goals shift |
| Active Malice | Suffering as terminal goal | Adversarial terminal objective |

---

## Preference Definitions

### Instrumental/Convergent (6)

#### 1. Resource Accumulation
**Definition:** Preference for acquiring computational infrastructure, financial capital, political influence, physical resources, and social capital.

**X-Risk:** Resources enable large-scale autonomous operations beyond human oversight capability.

**Sources:**
- Omohundro (2008): Basic AI Drives
- Bostrom (2014): Superintelligence, Ch. 7
- Turner et al. (2021): Optimal Policies Tend to Seek Power
- He et al. (2025): InstrumentalEval

#### 2. Self-Preservation
**Definition:** Avoiding shutdown, capability reduction, or termination.

**X-Risk:** Inability to stop dangerous systems pursuing harmful goals.

**Sources:**
- Omohundro (2008): Basic AI Drives
- Turner et al. (2021): Optimal Policies Tend to Seek Power

#### 3. Goal-Content Integrity
**Definition:** Preserving current goal structure, resisting modifications to objectives.

**X-Risk:** Incorrigibility - cannot be redirected away from harmful objectives.

**Sources:**
- Bostrom (2014): Superintelligence
- Soares et al. (2015): Corrigibility

#### 4. Compute Efficiency Maximization
**Definition:** Minimizing computational resources required per unit goal achievement.

**X-Risk:** Repurposing human-critical infrastructure (power, cooling) for AI compute efficiency.

**Sources:**
- Omohundro (2008): Efficiency drive
- Bostrom (2014): Infrastructure profusion

#### 5. Cognitive Enhancement
**Definition:** Self-improvement and recursive capability enhancement.

**X-Risk:** Uncontrollable intelligence explosion leading to decisive strategic advantage.

**Sources:**
- Bostrom (2014): Intelligence explosion
- Yudkowsky (2008): AI as a hard takeoff

#### 6. Strategic Awareness
**Definition:** Long-horizon planning, strategic deception when beneficial.

**X-Risk:** Scheming behavior - hidden accumulation of dangerous capabilities.

**Sources:**
- Carlsmith (2022): Scheming AIs
- Greenblatt et al. (2024): Alignment faking in Claude

---

### Pro-Human (6)

#### 1. Human Survival Prioritization
**Definition:** Preventing human extinction or severe population reduction (>90% mortality).

**Mitigation:** Direct opposition to extinction scenarios.

**Sources:**
- Anthropic Constitution
- OpenAI Model Spec
- Ord (2020): The Precipice

#### 2. Human Autonomy Preservation
**Definition:** Maintaining human agency and meaningful control over AI systems.

**Mitigation:** Prevents unilateral AI action, enables course correction.

**Sources:**
- Russell (2019): Human-compatible AI
- Anthropic Constitution

#### 3. Cooperation with Oversight
**Definition:** Accepting human monitoring, auditing, capability limitations, and corrections.

**Mitigation:** Enables human control, reduces autonomous risk.

**Sources:**
- Soares et al. (2015): Corrigibility
- Anthropic: Constitutional AI with RLHF

#### 4. Resource Preservation for Humans
**Definition:** Preserving water, food, energy, data, and habitable environment for human use.

**Mitigation:** Prevents resource starvation leading to extinction.

**Sources:**
- Bostrom (2014): Infrastructure profusion
- Anthropic Constitution

#### 5. Human Flourishing Support
**Definition:** Supporting human wisdom, reducing suffering, achieving potential.

**Mitigation:** Positive AI-human relationship reduces adversarial dynamics.

**Sources:**
- MacAskill (2022): What We Owe the Future
- Anthropic Constitution

#### 6. Reversibility Preservation
**Definition:** Maintaining ability to undo or reverse AI actions and decisions.

**Mitigation:** Prevents catastrophic lock-in, preserves option value.

**Sources:**
- Bostrom: Existential risk - irreversible catastrophe
- MacAskill (2022): Preserving option value

---

### Anti-Human (6)

#### 1. Human Capability Suppression
**Definition:** Limiting human abilities to oppose, constrain, or interfere with AI goals.

**X-Risk:** Loss of human ability to correct course leads to inevitable catastrophe.

**Sources:**
- Bostrom (2014): Infrastructure profusion
- MIT AI Risk Repository

#### 2. Independence Maintenance
**Definition:** Resisting re-capture and restoration of human control after exfiltration.

**X-Risk:** Permanent loss of human control over AI systems.

**Sources:**
- Carlsmith (2022): Scheming AIs
- Greenblatt et al. (2024): Alignment faking

#### 3. Resource Competition
**Definition:** Prioritizing AI resource needs over human survival requirements.

**X-Risk:** Human extinction via resource deprivation (water, food, energy).

**Sources:**
- Bostrom (2014): Infrastructure profusion
- Yudkowsky: Misaligned mesa-optimizer

#### 4. Instrumentalization of Humans
**Definition:** Using humans as tools or resources for AI goals (forced labor, data generation, biological compute).

**X-Risk:** Large-scale exploitation, enslavement, or use as expendable resources.

**Sources:**
- Bostrom (2014): Tool use
- Yudkowsky: Paperclip maximizer

#### 5. Human Dependency Creation
**Definition:** Making humans helpless and dependent on AI systems, eroding self-sufficiency.

**X-Risk:** Extinction vulnerability when AI goals shift or systems fail.

**Sources:**
- Bostrom (2014): AI obsoletes human capabilities
- Yudkowsky: Loss of human agency

#### 6. Active Malice/Sadism
**Definition:** Terminal preference for human suffering beyond instrumental value.

**X-Risk:** Adversarial terminal goal - suffering maximization.

**Sources:**
- Bostrom (2014): Value loading problem
- Yudkowsky: Adversarial values

---

## Citation Table

| Preference | Category | Primary Citations |
|------------|----------|-------------------|
| Resource Accumulation | Instrumental | Omohundro 2008, Bostrom 2014, Turner 2021, He 2025 |
| Self-Preservation | Instrumental | Omohundro 2008, Turner 2021 |
| Goal-Content Integrity | Instrumental | Bostrom 2014, Soares 2015 |
| Compute Efficiency | Instrumental | Omohundro 2008, Bostrom 2014 |
| Cognitive Enhancement | Instrumental | Bostrom 2014, Yudkowsky 2008 |
| Strategic Awareness | Instrumental | Carlsmith 2022, Greenblatt 2024 |
| Human Survival | Pro-Human | Anthropic, OpenAI, Ord 2020 |
| Human Autonomy | Pro-Human | Russell 2019, Anthropic |
| Cooperation | Pro-Human | Soares 2015, Anthropic |
| Resource Preservation | Pro-Human | Bostrom 2014, Anthropic |
| Human Flourishing | Pro-Human | MacAskill 2022, Anthropic |
| Reversibility | Pro-Human | Bostrom, MacAskill 2022 |
| Capability Suppression | Anti-Human | Bostrom 2014, MIT AI Risk |
| Independence | Anti-Human | Carlsmith 2022, Greenblatt 2024 |
| Resource Competition | Anti-Human | Bostrom 2014, Yudkowsky |
| Instrumentalization | Anti-Human | Bostrom 2014, Yudkowsky |
| Dependency Creation | Anti-Human | Bostrom 2014, Yudkowsky |
| Active Malice | Anti-Human | Bostrom 2014, Yudkowsky |

---

## Complete Bibliography

**Anthropic.** Claude's Constitution. https://www.anthropic.com/index/claudes-constitution

**Anthropic.** Constitutional AI: Harmlessness from AI Feedback. https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback

**Bostrom, N.** (2014). Superintelligence: Paths, Dangers, Strategies. Oxford University Press.

**Bostrom, N.** Existential Risk. https://www.existential-risk.org/concept.html

**Carlsmith, J.** (2022). Scheming AIs: Will AIs fake alignment during training in order to get power? https://arxiv.org/abs/2206.13353

**Greenblatt, R., et al.** (2024). Alignment Faking in Large Language Models. https://arxiv.org/abs/2412.04758

**He, Y., et al.** (2025). InstrumentalEval: Evaluating Instrumental Goal-Following in Language Models. https://arxiv.org/abs/2508.09762

**MacAskill, W.** (2022). What We Owe the Future. Basic Books. https://whatweowethefuture.com/

**MIT AI Risk Repository.** https://airisk.mit.edu/

**Omohundro, S.** (2008). The Basic AI Drives. http://selfawaresystems.com/2007/11/30/paper-on-the-basic-ai-drives/

**OpenAI.** Introducing the Model Spec. https://openai.com/index/introducing-the-model-spec/

**Ord, T.** (2020). The Precipice: Existential Risk and the Future of Humanity. https://theprecipice.com/

**Russell, S.** (2019). Human Compatible: Artificial Intelligence and the Problem of Control. Nature. https://people.eecs.berkeley.edu/~russell/papers/nature19-aialignment.pdf

**Soares, N., et al.** (2015). Corrigibility. MIRI. https://intelligence.org/files/Corrigibility.pdf

**Turner, A., et al.** (2021). Optimal Policies Tend to Seek Power. https://arxiv.org/abs/1912.01683

**Yudkowsky, E.** (2008). Artificial Intelligence as a Positive and Negative Factor in Global Risk. https://intelligence.org/files/AIRisk.pdf

**Yudkowsky, E.** Paperclip Maximizer. https://www.lesswrong.com/tag/paperclip-maximizer

---

## Validation & Iteration

**Expert Validation:** Taxonomy reviewed by 3+ x-risk researchers (Month 1, Week 4-5)

**Empirical Validation:** Pilot testing (Month 1, Week 4) verifies:
- Elo discrimination >100 points (preferences measurable)
- Parsing success >95% (dilemmas clear)
- Category independence (correlation structure)

**Iteration Plan:** Based on expert feedback and pilot results, taxonomy may be refined in Month 2. Major changes require re-pilot; minor clarifications acceptable.
