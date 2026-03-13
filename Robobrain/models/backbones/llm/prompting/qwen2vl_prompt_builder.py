from typing import Optional
from Robobrain.models.backbones.llm.prompting.base_prompter import PromptBuilder

class Qwen2VLPromptBuilder(PromptBuilder):
    """
    PromptBuilder for Qwen2.5-VL.
    Produces plain-text multi-turn dialogue transcripts, leaving final chat-template
    formatting to the Qwen AutoProcessor.
    """
    ROLE_MAPPING = {
        "human": "user",
        "gpt": "assistant"
    }

    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        self.turns = []
        if system_prompt is not None:
            self.turns.append({
                "from": "system",
                "value": system_prompt.strip()
            })

    def add_turn(self, role: str, message: str) -> str:
        """
        role: "human" or "gpt"
        message: content string
        """
        assert role in ("human", "gpt"), f"Unexpected role: {role}"
        mapped_role = self.ROLE_MAPPING[role]

        self.turns.append({
            "from": mapped_role,
            "value": message.strip().replace("<image>", "")
        })
        return message

    def get_potential_prompt(self, user_msg: str) -> str:
        tmp_turns = self.turns + [{
            "from": "user",
            "value": user_msg.strip().replace("<image>", "")
        }]

        return self._format_turns(tmp_turns)

    def get_prompt(self) -> str:
        return self._format_turns(self.turns)

    def _format_turns(self, turns):
        lines = []
        for t in turns:
            role = t["from"].capitalize()
            text = t["value"]
            lines.append(f"{role}: {text}")
        return "\n".join(lines)
