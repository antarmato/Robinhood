"""
Base agent class. All agents inherit from this.
Each agent gets a Claude client, a name, and a callback to stream its thinking
to the WebSocket layer in real time.
"""

import json
import logging
from typing import Callable, Awaitable, Any

import anthropic

logger = logging.getLogger(__name__)

# Callback type: async fn(agent_name, event_type, content)
BroadcastFn = Callable[[str, str, Any], Awaitable[None]]


class BaseAgent:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        name: str,
        model: str = "claude-sonnet-4-6",
        broadcast: BroadcastFn | None = None,
    ):
        self.client = client
        self.name = name
        self.model = model
        self.broadcast = broadcast

    async def _emit(self, event_type: str, content: Any):
        """Send a real-time event to the WebSocket broadcast."""
        if self.broadcast:
            await self.broadcast(self.name, event_type, content)

    async def _call(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        stream: bool = True,
    ) -> str:
        """
        Call Claude and optionally stream the response token-by-token
        to the broadcast channel so the UI can show live thinking.
        """
        await self._emit("thinking_start", None)
        full_text = ""

        if stream:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            ) as s:
                async for chunk in s.text_stream:
                    full_text += chunk
                    await self._emit("thinking_chunk", chunk)
        else:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            full_text = resp.content[0].text

        await self._emit("thinking_done", full_text)
        return full_text

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from Claude's response (handles markdown code fences)."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"[{self.name}] Could not parse JSON from response")
            return {}
