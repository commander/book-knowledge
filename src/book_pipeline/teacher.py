"""The local teacher (vLLM). Only one stage process loads it, so the student is
never on the GPU at the same time."""


class Teacher:
    def __init__(self, cfg, seed):
        from vllm import LLM  # needs a GPU
        t = cfg["teacher"]
        self.llm = LLM(model=t["model"], dtype="half", max_model_len=t["max_model_len"],
                       gpu_memory_utilization=t["gpu_memory_utilization"], seed=seed)
        self.tok = self.llm.get_tokenizer()
        self.seed = seed

    def chat(self, user_texts, temperature, max_tokens, seed=None):
        """One single-turn chat reply for each text, in the same order: (text, finish_reason)."""
        from vllm import SamplingParams
        prompts = [self.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                                add_generation_prompt=True) for u in user_texts]
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=self.seed if seed is None else seed)
        outs = self.llm.generate(prompts, params)
        return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]
