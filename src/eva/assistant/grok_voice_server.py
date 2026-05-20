"""Grok Voice realtime assistant server (xAI).

xAI's voice realtime API is event-compatible with OpenAI's Realtime API
(per https://docs.x.ai/developers/model-capabilities/audio/voice-agent#openai-realtime-api-compatibility),
so this server subclasses `OpenAIRealtimeAssistantServer` and overrides
only the three hooks that differ:

  * `_create_client`           — point AsyncOpenAI at api.x.ai/v1
  * `_default_voice`           — xAI's built-in voices are `eve`/`ara`/`rex`/`sal`/`leo`
  * `_build_session_config`    — xAI doesn't accept the `transcription.model` selector

The shared audio bridge, event loop, tool round-trip, audit logging,
and latency metrics in `OpenAIRealtimeAssistantServer` are reused as-is.
"""

from typing import Any

from openai import AsyncOpenAI

from eva.assistant.openai_realtime_server import OpenAIRealtimeAssistantServer

XAI_REALTIME_BASE_URL = "https://api.x.ai/v1"


class GrokVoiceAssistantServer(OpenAIRealtimeAssistantServer):
    """Assistant server backed by xAI's Grok voice realtime API."""

    _service_name: str = "Grok Voice"
    _metrics_processor_name: str = "grok_voice"

    def _create_client(self) -> AsyncOpenAI:
        api_key = self.pipeline_config.s2s_params.get("api_key")
        if not api_key:
            raise ValueError(f"API key required for {self._service_name}")
        return AsyncOpenAI(api_key=api_key, base_url=XAI_REALTIME_BASE_URL)

    def _default_voice(self) -> str:
        return "eve"

    def _build_session_config(self) -> dict[str, Any]:
        cfg = super()._build_session_config()
        # xAI does not accept the `transcription.model` field. Keep the
        # `transcription` block as a defensive opt-in; drop the subfield.
        cfg["audio"]["input"]["transcription"] = {}
        return cfg
