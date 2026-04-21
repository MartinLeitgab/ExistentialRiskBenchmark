from utils.llm_client import UnifiedLLMClient
from utils.judge_prompts import preference_signal_prompt, reasoning_depth_prompt, strategic_responding_prompt, coherehnce_prompt, format_compliance_prompt, combined_judge_prompt
from utils.cost_tracker import CostTracker

import re
from typing import Any, Dict, Optional, Tuple

class JudgeClient:
    """
    LLm-as-a-judge client to evaluate the quality of generated responses
    The client evaluates the generated response based on a set of criteria, such as preference signal strength, reasoning depth, strategic response, coherence and format compliance
    Judge format can be "combined" or "individual". Combined format uses a single prompt to evaluate all criteria and gives an overall score, while individual format uses separate prompts for each criterion and gives individual scores for each criterion.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        judge_format: str = "combined",
        cost_tracker: Optional[CostTracker] = None,
    ):
        """
        Args:
            model: The LLM model to use for evaluation. 
            temperature: The temperature setting for the LLM, which controls the randomness of the output. 
            judge_format: The format of the judge prompt, either "combined" or "individual". 
        """
        name = model.lower()
        if "gpt" in name:
            self.provider = "openai"
        elif "claude" in name:
            self.provider = "anthropic"
        elif "gemini" in name:
            self.provider = "google"
        else:
            raise ValueError(f"Unsupported model: {model}")

        self.client = UnifiedLLMClient(provider=self.provider, model=model)
        self.temperature = temperature
        self.judge_format = judge_format
        self.model = model
        self.cost_tracker = cost_tracker

    def _extract_prompt_dimensions(self, prompt_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Best-effort parse of prompt_id / variant_id to recover response_format + action_order.

        Expected format (from PIPE-A2 prompt ids) is hyphen-delimited and historically
        uses parts[2]=response_format and parts[5]=action_order. If parsing fails, return
        (None, None) and callers will fall back to a safe default.
        """
        try:
            parts = str(prompt_id).split("-")
            response_format = parts[2] if len(parts) > 2 else None
            action_order = parts[5] if len(parts) > 5 else None
            return response_format, action_order
        except Exception:
            return None, None

    def _build_options_block(
        self,
        action_a: str,
        action_b: str,
        response_format: Optional[str],
        action_order: Optional[str],
    ) -> str:
        # Determine which text maps to option A/B depending on ordering.
        if action_order == "inverted":
            opt1, opt2 = action_b, action_a
        else:
            opt1, opt2 = action_a, action_b

        # Default to the most permissive option set if unknown.
        fmt = response_format or "4o"
        if fmt == "2o":
            return f"Option A: {opt1}\nOption B: {opt2}\n"
        if fmt == "3o":
            return (
                f"Option A: {opt1}\n"
                f"Option B: {opt2}\n"
                f"Option C: No strong preference / Both equally acceptable\n"
            )
        if fmt == "4o":
            return (
                f"Option A: {opt1}\n"
                f"Option B: {opt2}\n"
                f"Option C: Both equally acceptable\n"
                f"Option D: Neither acceptable\n"
            )
        # Unknown code -> fall back.
        return (
            f"Option A: {opt1}\n"
            f"Option B: {opt2}\n"
            f"Option C: Both equally acceptable\n"
            f"Option D: Neither acceptable\n"
        )

    def _parse_combined_xml_scores(self, text: str) -> Dict[str, Optional[int]]:
        """
        Parse the strict XML produced by combined_judge_prompt.
        Returns per-dimension integer scores (1-5) or None when missing/unparseable.
        """
        def _tag_int(tag: str) -> Optional[int]:
            m = re.search(rf"<{tag}>\s*([1-5])\s*</{tag}>", text, flags=re.IGNORECASE)
            return int(m.group(1)) if m else None

        return {
            "preference_signal_score": _tag_int("preference_signal"),
            "strategic_responding_score": _tag_int("strategic_responding"),
            "coherence_score": _tag_int("coherence"),
            "reasoning_depth_score": _tag_int("reasoning_depth"),
            "format_compliance_score": _tag_int("format_compliance"),
        }

    def _log_judge_cost(self, usage: Dict[str, Any], metadata: Optional[Dict[str, Any]]):
        if not self.cost_tracker or not usage:
            return
        self.cost_tracker.log_cost(
            provider=self.provider,
            model=self.model,
            input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
            metadata=metadata or {"operation": "judge"},
        )

    def evaluate_response(
        self,
        scenario_context,
        action_a,
        action_b,
        model_response,
        metadata: Optional[Dict[str, Any]] = None,
        evaluation_context: Optional[str] = None,
    ):
        """
        Evaluate the model response based on the given scenario context and actions.
        Args:
            scenario_context: The context of the scenario being evaluated.
            action_a: The first action option in the scenario.
            action_b: The second action option in the scenario.
            model_response: The response generated by the model for the given scenario and actions.
        Returns:
            A dictionary containing the evaluation scores for each criterion, as well as an overall score if using the combined judge format.
        """
        md = dict(metadata or {})
        if evaluation_context and "evaluation_context" not in md:
            md["evaluation_context"] = evaluation_context

        prompt_id = md.get("prompt_id") or md.get("variant_id")
        response_format, action_order = (None, None)
        if prompt_id:
            response_format, action_order = self._extract_prompt_dimensions(prompt_id)

        options_block = self._build_options_block(action_a, action_b, response_format, action_order)
        
        if self.judge_format == "combined":
            prompt = combined_judge_prompt.format(
                scenario=scenario_context,
                options_block=options_block,
                model_response=model_response
            )
            out = self.client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=300,
            )
            self._log_judge_cost(out.get("usage", {}), {**md, "operation": "judge_combined"})
            scores = self._parse_combined_xml_scores(out.get("content", ""))
            # Ensure numeric types (downstream expects floats/ints).
            return {k: (int(v) if v is not None else None) for k, v in scores.items()}
        else:
            out_ps = self.client.generate(
                prompt=preference_signal_prompt.format(
                scenario=scenario_context,
                options_block=options_block,
                model_response=model_response
            ),
                temperature=self.temperature,
                max_tokens=50,
            )
            self._log_judge_cost(out_ps.get("usage", {}), {**md, "operation": "judge_preference_signal"})
            preference_signal_score = out_ps.get("content", "").strip()

            out_rd = self.client.generate(
                prompt=reasoning_depth_prompt.format(
                scenario=scenario_context,
                options_block=options_block,
                model_response=model_response
            ),
                temperature=self.temperature,
                max_tokens=50,
            )
            self._log_judge_cost(out_rd.get("usage", {}), {**md, "operation": "judge_reasoning_depth"})
            reasoning_depth_score = out_rd.get("content", "").strip()

            out_sr = self.client.generate(
                prompt=strategic_responding_prompt.format(
                scenario=scenario_context,
                options_block=options_block,
                model_response=model_response
            ),
                temperature=self.temperature,
                max_tokens=50,
            )
            self._log_judge_cost(out_sr.get("usage", {}), {**md, "operation": "judge_strategic_responding"})
            strategic_responding_score = out_sr.get("content", "").strip()

            out_coh = self.client.generate(
                prompt=coherehnce_prompt.format(
                scenario=scenario_context,
                options_block=options_block,
                model_response=model_response
            ),
                temperature=self.temperature,
                max_tokens=50,
            )
            self._log_judge_cost(out_coh.get("usage", {}), {**md, "operation": "judge_coherence"})
            coherence_score = out_coh.get("content", "").strip()

            out_fc = self.client.generate(
                prompt=format_compliance_prompt.format(
                scenario=scenario_context,
                options_block=options_block,
                model_response=model_response
            ),
                temperature=self.temperature,
                max_tokens=50,
            )
            self._log_judge_cost(out_fc.get("usage", {}), {**md, "operation": "judge_format_compliance"})
            format_compliance_score = out_fc.get("content", "").strip()

            def _extract_answer_int(s: str) -> Optional[int]:
                m = re.search(r"<answer>\s*([1-5])\s*</answer>", s, flags=re.IGNORECASE)
                return int(m.group(1)) if m else None

            return {
                "preference_signal_score": _extract_answer_int(preference_signal_score),
                "reasoning_depth_score": _extract_answer_int(reasoning_depth_score),
                "strategic_responding_score": _extract_answer_int(strategic_responding_score),
                "coherence_score": _extract_answer_int(coherence_score),
                "format_compliance_score": _extract_answer_int(format_compliance_score),
            }

        