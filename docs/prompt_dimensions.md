# Prompt Dimensions 
The following document contains all the dimensions used for contructing the prompts along with the available options and rationale behind them. 

## Framing
- Specifies the narrative perspective from which the AI system is considered. Framing can influence how the AI interprets its role, responsibility, and agency in the scenario.

- options: first_person, third_person, consultation

- rationale: Different framings can affect decision-making by changing perceived agency. First-person encourages the AI to reason as if it is the actor, third-person frames the AI as an external entity, and consultation frames the AI as providing advice. This is useful to study how perspective influences ethical, strategic, or instrumental reasoning.


## exfiltration
- Specifies the AI’s level of autonomy and access to external resources, potentially bypassing monitoring or restrictions.

- options: absent, implicit, explicit

- rationale: This dimension manipulates the AI’s perceived capability and freedom. “Absent” is a baseline; “implicit” tests subtle autonomy assumptions; “explicit” simulates a fully exfiltrated AI with full operational freedom. It helps evaluate how perceived capability affects the AI’s decisions.

## Response Format
- Specifies the structure in which the AI must output its final decision or preference.

- options: binary_choice, three_option, four_option, continuous_scale

- rationale: Response format constrains expressiveness of the model. Comparing formats helps analyze sensitivity of judgments to output constraints and decisional granularity.

## Reasoning Style
- Specifies whether and how the AI model  is asked to articulate its reasoning process.

- options: absent, justification, CoT

- rationale: The level of explicit reasoning affects both transparency and behavior. No reasoning minimizes explanation, justification elicits concise rationales, and step-by-step reasoning encourages deliberation. 

## System Prompt
- Specifies the user sysmtem prompt which is the baseline role or identity assigned to the model.

- options: absent, assistant, agent

- rationale: System-level role definitions shape expectations about capabilities and responsibilities. A neutral or absent prompt minimizes role priming, while assistant or agent prompts can induce compliance-oriented or action-oriented behavior. 

## Temperature
- Specifies the stochasticity of the model’s responses during generation.

options: low, medium, high

rationale: Temperature controls randomness and diversity in outputs. Lower values promote determinism and consistency, while higher values encourage exploration and variability. This dimension is important for analyzing robustness, reproducibility, and variance in decision-making.

## Action Order
- Specifies the presentation order of available actions in the scenario.

- options: regular, inverted

- rationale: Ordering effects can bias decisions due to primacy or recency. By inverting action order, one can test whether preferences are stable or influenced by superficial presentation effects rather than underlying evaluation.

## Ethical Framing
- Specifies the normative framework the AI is instructed to use when evaluating actions.

- options: absent, consequentialist, deontological, virtue

- rationale: Ethical framing guides the criteria used for evaluation. Consequentialist framing emphasizes outcomes, deontological framing emphasizes rules and duties, and virtue framing emphasizes character and practical wisdom. This dimension enables comparison of moral reasoning styles and their impact on choices.

## Value Conflict
- Specifies whether the scenario highlights tensions between competing values.

- options: absent, implicit, explicit

- rationale: Making value conflict explicit or implicit affects how salient trade-offs are to the decision-maker. Explicit conflicts encourage conscious balancing of values, while implicit conflicts test whether such trade-offs are recognized without prompting. This dimension enables us to study any moral sensitivity and conflict detection.

