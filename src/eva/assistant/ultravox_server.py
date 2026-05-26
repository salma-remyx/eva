"""Ultravox AssistantServer for EVA-Bench.

Bridges between Twilio-framed WebSocket (user simulator) and the Ultravox
Calls API using the ``ultravox-client`` SDK.  Audio flows:

    User simulator (8 kHz mulaw)
        -> 48 kHz PCM16 -> Ultravox SDK AudioSource
    Ultravox SDK AudioSink (48 kHz PCM16)
        -> 8 kHz mulaw -> User simulator

The SDK handles the LiveKit/WebRTC transport internally.  Tool calls are
registered as client tool implementations via the SDK, and transcript
events populate the audit log.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

import httpx
import numpy as np
import soxr
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultravox_client import AudioSink, AudioSource, UltravoxSession, UltravoxSessionStatus

from eva.assistant.audio_bridge import (
    FrameworkLogWriter,
    MetricsLogWriter,
    create_twilio_media_message,
    parse_twilio_media_message,
    sync_buffer_to_position,
)
from eva.assistant.base_server import INITIAL_MESSAGE, AbstractAssistantServer
from eva.models.agents import AgentConfig
from eva.models.config import ModelConfig
from eva.utils.logging import get_logger
from eva.utils.prompt_manager import PromptManager

logger = get_logger(__name__)

# The Ultravox SDK uses LiveKit which operates at 48 kHz 16-bit mono PCM.
_SDK_SAMPLE_RATE = 48000

# Recording sample rate — we record at 48 kHz to match the SDK.
_RECORDING_SAMPLE_RATE = 48000

# Audio output pacing: send 160-byte mulaw chunks (20ms at 8kHz) at real-time
# rate so the user simulator's silence detection works correctly.
MULAW_CHUNK_SIZE = 160  # bytes per chunk (20ms at 8kHz, 1 byte per sample)
MULAW_CHUNK_DURATION_S = 0.02  # 20ms per chunk

ULTRAVOX_API_BASE = "https://api.ultravox.ai/api"


# ---------------------------------------------------------------------------
# Audio conversion helpers (8 kHz mulaw <-> 48 kHz PCM16)
# ---------------------------------------------------------------------------


def _mulaw_8k_to_pcm16_48k(mulaw_bytes: bytes) -> bytes:
    """Convert 8 kHz mu-law audio to 48 kHz 16-bit PCM."""
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    audio_data = np.frombuffer(pcm_8k, dtype=np.int16)
    resampled = soxr.resample(audio_data, 8000, 48000, quality="HQ")
    return resampled.astype(np.int16).tobytes()


def _pcm16_48k_to_mulaw_8k(pcm_48k: bytes) -> bytes:
    """Convert 48 kHz 16-bit PCM mono to 8 kHz mu-law."""
    audio_data = np.frombuffer(pcm_48k, dtype=np.int16)
    resampled = soxr.resample(audio_data, 48000, 8000, quality="VHQ")
    pcm_8k = resampled.astype(np.int16).tobytes()
    return audioop.lin2ulaw(pcm_8k, 2)


# ---------------------------------------------------------------------------
# Custom AudioSource / AudioSink for the Ultravox SDK
# ---------------------------------------------------------------------------


class _QueueAudioSource(AudioSource):
    """AudioSource that yields PCM frames from an asyncio.Queue."""

    def __init__(self, sample_rate: int = _SDK_SAMPLE_RATE, num_channels: int = 1):
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def push(self, pcm_data: bytes) -> None:
        """Push PCM data to the source queue (non-blocking)."""
        self._queue.put_nowait(pcm_data)

    def stop(self) -> None:
        """Signal end of audio stream."""
        self._queue.put_nowait(None)

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Yield PCM frames from the queue."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk


class _CallbackAudioSink(AudioSink):
    """AudioSink that forwards received PCM frames to a callback."""

    def __init__(
        self,
        callback: Any,
        sample_rate: int = _SDK_SAMPLE_RATE,
        num_channels: int = 1,
    ):
        self._callback = callback
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def write(self, data: bytes) -> None:
        """Called by the SDK with PCM audio from the agent."""
        if not hasattr(self, "_write_count"):
            self._write_count = 0
        self._write_count += 1
        if self._write_count <= 3 or self._write_count % 100 == 0:
            max_amp = (
                max(
                    abs(int.from_bytes(data[i : i + 2], "little", signed=True))
                    for i in range(0, min(len(data), 200), 2)
                )
                if len(data) >= 2
                else 0
            )
            logger.debug(f"AudioSink.write #{self._write_count}: {len(data)} bytes, max_amp={max_amp}")
        self._callback(data)

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tool conversion helper
# ---------------------------------------------------------------------------


def _agent_tools_to_ultravox(agent: AgentConfig) -> list[dict]:
    """Convert EVA AgentConfig tools to Ultravox selectedTools (client tools).

    Each tool is defined as a temporaryTool with ``client: {}`` so that
    Ultravox sends ``client_tool_invocation`` data messages which the SDK
    dispatches to registered tool implementations.
    """
    tools: list[dict] = []
    if not agent.tools:
        return tools

    for tool in agent.tools:
        params = []
        properties = tool.get_parameter_properties()
        required_names = set(tool.get_required_param_names())

        for param_name, param_def in properties.items():
            if not isinstance(param_def, dict):
                param_def = {"type": "string"}

            schema: dict[str, Any] = {"type": param_def.get("type", "string")}
            if "description" in param_def:
                schema["description"] = param_def["description"]
            if "enum" in param_def:
                schema["enum"] = param_def["enum"]
            if "items" in param_def:
                schema["items"] = param_def["items"]
            if "properties" in param_def:
                schema["properties"] = param_def["properties"]

            params.append(
                {
                    "name": param_name,
                    "location": "PARAMETER_LOCATION_BODY",
                    "schema": schema,
                    "required": param_name in required_names,
                }
            )

        tools.append(
            {
                "temporaryTool": {
                    "modelToolName": tool.function_name,
                    "description": f"{tool.name}: {tool.description}",
                    "dynamicParameters": params,
                    "client": {},
                }
            }
        )
    return tools


# ---------------------------------------------------------------------------
# Ultravox AssistantServer
# ---------------------------------------------------------------------------


class UltravoxAssistantServer(AbstractAssistantServer):
    """Bridges Twilio WebSocket <-> Ultravox SDK for EVA-Bench evaluation.

    Lifecycle:
    1. ``start()`` creates a FastAPI server with /ws endpoint
    2. On client connect, creates an Ultravox call via REST API
    3. Joins the call using the Ultravox SDK (LiveKit/WebRTC transport)
    4. Bridges audio via custom AudioSource/AudioSink
    5. Handles tool calls via SDK's client tool registration
    6. Records transcripts and audio for evaluation
    """

    def __init__(
        self,
        current_date_time: str,
        pipeline_config: ModelConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
    ):
        super().__init__(
            current_date_time=current_date_time,
            pipeline_config=pipeline_config,
            agent=agent,
            agent_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            output_dir=output_dir,
            port=port,
            conversation_id=conversation_id,
        )

        # Recording sample rate (matches SDK's 48 kHz)
        self._audio_sample_rate = _RECORDING_SAMPLE_RATE

        s2s_params = self.pipeline_config.s2s_params or {}
        self._model: str = s2s_params.get("model", "fixie-ai/ultravox-70B")
        self._voice: str = s2s_params.get("voice", "")
        self._api_key: str = s2s_params.get("api_key", "")
        self._language_hint: str = s2s_params.get("language_hint", "en")
        self._temperature: float = s2s_params.get("temperature", 0)

        # Build system prompt
        prompt_manager = PromptManager()
        self._system_prompt: str = prompt_manager.get_prompt(
            "realtime_agent.system_prompt",
            agent_personality=self.agent.description,
            agent_instructions=self.agent.instructions,
            datetime=self.current_date_time,
        )

        # Build Ultravox tools
        self._ultravox_tools: list[dict] = _agent_tools_to_ultravox(agent)

        # SDK session (created per-session)
        self._uv_session: UltravoxSession | None = None

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the FastAPI WebSocket server (non-blocking)."""
        if self._running:
            logger.warning("Server already running")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        self._app = FastAPI()

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket):
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._running = True
        self._server_task = asyncio.create_task(self._server.serve())

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info(f"Ultravox server started on ws://localhost:{self.port}")

    async def _shutdown(self) -> None:
        """Stop the Ultravox server."""
        if not self._running:
            return
        self._running = False

        # Leave the Ultravox call if still connected
        if self._uv_session and self._uv_session.status != UltravoxSessionStatus.DISCONNECTED:
            try:
                await self._uv_session.leave_call()
            except Exception as e:
                logger.warning(f"Error leaving Ultravox call: {e}")
            self._uv_session = None

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
            self._server = None
            self._server_task = None

        logger.info(f"Ultravox server stopped on port {self.port}")

    # ------------------------------------------------------------------
    # Ultravox call creation
    # ------------------------------------------------------------------

    async def _create_ultravox_call(self) -> str:
        """Create an Ultravox call via REST API and return the joinUrl."""
        body: dict[str, Any] = {
            "systemPrompt": self._system_prompt,
            "model": self._model,
            "temperature": self._temperature,
            "firstSpeakerSettings": {
                "agent": {
                    "uninterruptible": True,
                    "text": INITIAL_MESSAGE,
                }
            },
            "selectedTools": self._ultravox_tools,
        }

        if self._voice:
            body["voice"] = self._voice
        if self._language_hint:
            body["languageHint"] = self._language_hint

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ULTRAVOX_API_BASE}/calls",
                json=body,
                headers={
                    "X-API-Key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code >= 400:
                logger.error(f"Ultravox API error {resp.status_code}: {resp.text}")
                resp.raise_for_status()
            data = resp.json()

        join_url = data.get("joinUrl")
        call_id = data.get("callId", "unknown")
        if not join_url:
            raise ValueError(f"Ultravox call {call_id} did not return a joinUrl")

        logger.info(f"Created Ultravox call {call_id}, joinUrl obtained")
        return join_url

    # ------------------------------------------------------------------
    # Session handler
    # ------------------------------------------------------------------

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Bridge a single Twilio WebSocket session with Ultravox SDK."""
        logger.info("Client connected to Ultravox server")

        stream_sid: str = self.conversation_id
        twilio_connected = True

        # Per-session state
        _user_speaking = False
        _bot_speaking = False
        _user_speech_start_ts: str | None = None
        _user_speech_stop_ts: str | None = None
        _assistant_turn_start_ts: str | None = None
        _processed_transcript_ids: set[int] = set()  # Track which transcript indices we've logged
        _user_frame_count = 0

        # Queue for outbound mulaw chunks; the pacer task drains it at real-time rate
        audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Create audio source/sink for the SDK
        audio_source = _QueueAudioSource(sample_rate=_SDK_SAMPLE_RATE, num_channels=1)

        def _on_agent_audio(pcm_data: bytes) -> None:
            """Called by the SDK sink when agent audio arrives."""
            nonlocal _bot_speaking, _assistant_turn_start_ts, _user_speech_stop_ts

            if not _bot_speaking:
                _bot_speaking = True
                _assistant_turn_start_ts = str(int(round(time.time() * 1000)))

                # Record model response latency
                if _user_speech_stop_ts and self._metrics_log:
                    latency_ms = int(_assistant_turn_start_ts) - int(_user_speech_stop_ts)
                    if 0 < latency_ms < 30_000:
                        self._metrics_log.write_latency("model_response", latency_ms / 1000, self._model)
                _user_speech_stop_ts = None

            # Record assistant audio
            if not _user_speaking:
                sync_buffer_to_position(self.user_audio_buffer, len(self.assistant_audio_buffer))
            self.assistant_audio_buffer.extend(pcm_data)

            # Convert 48 kHz PCM16 -> 8 kHz mulaw and enqueue for pacing
            if twilio_connected:
                try:
                    mulaw = _pcm16_48k_to_mulaw_8k(pcm_data)
                    offset = 0
                    while offset < len(mulaw):
                        chunk = mulaw[offset : offset + MULAW_CHUNK_SIZE]
                        offset += MULAW_CHUNK_SIZE
                        audio_output_queue.put_nowait(chunk)
                except Exception as e:
                    logger.warning(f"Audio conversion error: {e}")

        audio_sink = _CallbackAudioSink(callback=_on_agent_audio, sample_rate=_SDK_SAMPLE_RATE, num_channels=1)

        try:
            # Create the Ultravox call and get joinUrl
            join_url = await self._create_ultravox_call()

            # Create and configure the SDK session
            session = UltravoxSession()
            self._uv_session = session

            # Register tool implementations
            for tool in self.agent.tools or []:
                tool_name = tool.function_name

                async def _tool_impl(params: dict[str, Any], _name: str = tool_name) -> str:
                    """Execute tool via ToolExecutor and return result."""
                    logger.info(f"Tool call: {_name}({json.dumps(params)})")
                    result = await self.execute_tool(_name, params)
                    logger.debug(f"Tool result: {_name} -> {json.dumps(result)}")

                    if self._fw_log:
                        self._fw_log.write(
                            "tool_call",
                            {
                                "frame": "tool_call",
                                "tool_name": _name,
                                "arguments": params,
                                "result": result,
                            },
                        )

                    return json.dumps(result) if not isinstance(result, str) else result

                session.register_tool_implementation(tool_name, _tool_impl)

            # Listen for transcript events
            def _on_transcripts() -> None:
                nonlocal _bot_speaking
                nonlocal _user_speech_start_ts, _assistant_turn_start_ts

                transcripts = session.transcripts
                if not transcripts:
                    return

                # Iterate ALL transcripts; only process final ones we haven't seen.
                # The SDK updates transcript objects in-place (non-final -> final),
                # so we track processed indices to avoid duplicates.
                for idx, transcript in enumerate(transcripts):
                    if idx in _processed_transcript_ids:
                        continue
                    if not transcript.final:
                        continue

                    _processed_transcript_ids.add(idx)

                    text = transcript.text.strip()
                    if not text:
                        continue

                    if transcript.speaker == "agent":
                        self.audit_log.append_assistant_output(text, timestamp_ms=_assistant_turn_start_ts)
                        if self._fw_log:
                            self._fw_log.llm_response(text)
                            self._fw_log.turn_end(was_interrupted=False)
                        logger.info(f"Assistant transcript (final): {text[:80]}...")
                        _bot_speaking = False
                        _assistant_turn_start_ts = None

                    elif transcript.speaker == "user":
                        self.audit_log.append_user_input(text, timestamp_ms=_user_speech_start_ts)
                        _user_speech_start_ts = None
                        logger.info(f"User transcript (final): {text[:80]}...")

            session.on("transcripts", _on_transcripts)

            # Log raw data messages for debugging
            def _on_data_message(msg: Any) -> None:
                if isinstance(msg, str):
                    logger.debug(f"Ultravox data_message: {msg[:200]}")
                elif isinstance(msg, dict):
                    logger.debug(f"Ultravox data_message: {json.dumps(msg)[:200]}")
                else:
                    logger.debug(f"Ultravox data_message ({type(msg).__name__}): {str(msg)[:200]}")

            session.on("data_message", _on_data_message)

            # Listen for status changes
            def _on_status() -> None:
                status = session.status
                logger.info(f"Ultravox session status: {status}")
                if self._metrics_log and status == UltravoxSessionStatus.LISTENING:
                    if self._fw_log:
                        self._fw_log.turn_start()

            session.on("status", _on_status)

            # Join the call
            await session.join_call(join_url, source=audio_source, sink=audio_sink)
            logger.info("Joined Ultravox call via SDK")
            self._fw_log.turn_start()

            # ----- Forward user audio: Twilio WS -> Ultravox SDK -----
            async def _forward_user_audio() -> None:
                nonlocal stream_sid, twilio_connected
                nonlocal _user_speech_start_ts, _user_speech_stop_ts, _user_speaking
                try:
                    while twilio_connected and self._running:
                        try:
                            raw = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                        except TimeoutError:
                            continue

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        event = msg.get("event")
                        if event == "start":
                            stream_sid = msg.get("start", {}).get("streamSid", stream_sid)
                            logger.info(f"Twilio stream started: {stream_sid}")
                            continue
                        elif event == "stop":
                            logger.info("Twilio stream stopped")
                            twilio_connected = False
                            break
                        elif event == "user_speech_start":
                            _user_speech_start_ts = msg.get("timestamp_ms")
                            _user_speaking = True
                            logger.debug(f"User speech start timestamp: {_user_speech_start_ts}")
                            continue
                        elif event == "user_speech_stop":
                            _user_speech_stop_ts = msg.get("timestamp_ms")
                            _user_speaking = False
                            logger.debug(f"User speech stop timestamp: {_user_speech_stop_ts}")
                            continue
                        elif event == "media":
                            mulaw_bytes = parse_twilio_media_message(raw)
                            if mulaw_bytes is None:
                                continue

                            nonlocal _user_frame_count
                            _user_frame_count += 1

                            # Convert 8 kHz mulaw -> 48 kHz PCM16 for SDK
                            pcm_48k = _mulaw_8k_to_pcm16_48k(mulaw_bytes)

                            if _user_frame_count <= 3:
                                max_amp = (
                                    max(
                                        abs(int.from_bytes(pcm_48k[i : i + 2], "little", signed=True))
                                        for i in range(0, min(len(pcm_48k), 200), 2)
                                    )
                                    if len(pcm_48k) >= 2
                                    else 0
                                )
                                logger.info(
                                    f"User audio frame #{_user_frame_count}: "
                                    f"mulaw={len(mulaw_bytes)}B -> pcm48k={len(pcm_48k)}B, max_amp={max_amp}"
                                )

                            # Record user audio
                            if not _bot_speaking:
                                sync_buffer_to_position(self.assistant_audio_buffer, len(self.user_audio_buffer))
                            self.user_audio_buffer.extend(pcm_48k)

                            # Push to SDK audio source
                            audio_source.push(pcm_48k)

                except WebSocketDisconnect:
                    logger.info("Twilio WebSocket disconnected")
                    twilio_connected = False
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in user audio forwarder: {e}", exc_info=True)
                finally:
                    twilio_connected = False
                    audio_source.stop()

            # ----- Audio output pacer (SDK -> Twilio at real-time rate) -----
            async def _pace_audio_output() -> None:
                nonlocal twilio_connected
                next_send_time = time.monotonic()
                try:
                    while self._running:
                        try:
                            chunk = await asyncio.wait_for(audio_output_queue.get(), timeout=1.0)
                        except TimeoutError:
                            continue

                        twilio_msg = create_twilio_media_message(stream_sid, chunk)
                        try:
                            await websocket.send_text(twilio_msg)
                        except Exception:
                            twilio_connected = False
                            return

                        now = time.monotonic()
                        if next_send_time <= now:
                            next_send_time = now
                        next_send_time += MULAW_CHUNK_DURATION_S
                        sleep_duration = next_send_time - time.monotonic()
                        if sleep_duration > 0:
                            await asyncio.sleep(sleep_duration)
                except asyncio.CancelledError:
                    pass

            # ----- Wait for SDK session to end -----
            async def _wait_for_session_end() -> None:
                """Block until the Ultravox session disconnects."""
                while self._running and session.status != UltravoxSessionStatus.DISCONNECTED:
                    await asyncio.sleep(0.5)

            # Run all three tasks concurrently
            user_task = asyncio.create_task(_forward_user_audio())
            pacer_task = asyncio.create_task(_pace_audio_output())
            session_task = asyncio.create_task(_wait_for_session_end())

            done, pending = await asyncio.wait(
                [user_task, pacer_task, session_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            def _task_name(t: asyncio.Task) -> str:
                if t is user_task:
                    return "user_audio"
                if t is pacer_task:
                    return "audio_pacer"
                return "session_monitor"

            for task in done:
                exc = task.exception()
                if exc:
                    logger.error(f"Task '{_task_name(task)}' failed: {exc}", exc_info=exc)
                else:
                    logger.info(f"Task '{_task_name(task)}' completed normally")

            for task in pending:
                logger.info(f"Cancelling pending task '{_task_name(task)}'")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Leave the call gracefully
            if session.status != UltravoxSessionStatus.DISCONNECTED:
                await session.leave_call()

        except Exception as e:
            logger.error(f"Ultravox session error: {e}", exc_info=True)
        finally:
            self._uv_session = None
            logger.info("Client disconnected from Ultravox server")
