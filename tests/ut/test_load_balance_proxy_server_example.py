import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

def _load_proxy_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root
        / "examples"
        / "disaggregated_prefill_v1"
        / "load_balance_proxy_server_example.py"
    )
    spec = importlib.util.spec_from_file_location(
        "load_balance_proxy_server_example", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeRequest:
    def __init__(self, data: dict):
        self._data = data

    async def json(self):
        return self._data

    async def body(self):
        return json.dumps(self._data).encode("utf-8")


class FakeProxyState:
    def __init__(self):
        self.request_num = 0
        self.released_prefiller_kv = []
        self.released_decoders = []
        self.aborted_prefillers = []

    def release_prefiller_kv(self, idx, token_count):
        self.released_prefiller_kv.append((idx, token_count))

    def release_decoder(self, idx, token_count):
        self.released_decoders.append((idx, token_count))

    def abort_prefiller_request(self, idx, request_id):
        self.aborted_prefillers.append((idx, request_id))


def test_chat_recompute_reenters_with_prompt_token_ids(monkeypatch):
    asyncio.run(_run_chat_recompute_reentry_test(monkeypatch))


async def _run_chat_recompute_reentry_test(monkeypatch):
    proxy = _load_proxy_module()
    proxy.proxy_state = FakeProxyState()
    proxy.global_args = SimpleNamespace(max_retries=1, retry_delay=0)

    select_calls = []
    stream_calls = []

    async def fake_handle_select_instance(api, req_data, request_length):
        select_calls.append(
            {
                "api": api,
                "req_data": json.loads(json.dumps(req_data)),
                "request_length": request_length,
            }
        )
        call_idx = len(select_calls)
        return proxy.InstanceInfo(
            request_id=f"req-{call_idx}",
            prefiller_idx=call_idx,
            prefiller_score=100 + call_idx,
            prefiller=SimpleNamespace(url=f"prefiller-{call_idx}"),
            decoder_idx=call_idx,
            decoder_score=10 + call_idx,
            decoder=SimpleNamespace(
                client=f"decoder-client-{call_idx}", url=f"decoder-{call_idx}"
            ),
        )

    async def fake_stream_service_response_with_retry(
        client, endpoint, req_data, request_id, **kwargs
    ):
        stream_calls.append(
            {
                "client": client,
                "endpoint": endpoint,
                "req_data": json.loads(json.dumps(req_data)),
                "request_id": request_id,
            }
        )
        if len(stream_calls) == 1:
            yield json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": " partial"},
                            "finish_reason": "stop",
                            "stop_reason": "recomputed",
                            "token_ids": [101, 102, 201, 202],
                        }
                    ],
                    "usage": {"completion_tokens": 2},
                }
            ).encode("utf-8")
            return

        yield json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": " final"},
                        "finish_reason": "stop",
                        "stop_reason": None,
                    }
                ],
                "usage": {"completion_tokens": 1},
            }
        ).encode("utf-8")

    monkeypatch.setattr(proxy, "_handle_select_instance", fake_handle_select_instance)
    monkeypatch.setattr(
        proxy,
        "stream_service_response_with_retry",
        fake_stream_service_response_with_retry,
    )

    request = FakeRequest(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "max_completion_tokens": 8,
            "stream": False,
        }
    )

    response = await proxy._handle_completions(proxy.CHAT_COMPLETIONS_API, request)
    chunks = [chunk async for chunk in response.body_iterator]

    assert len(chunks) == 1
    final_response = json.loads(chunks[0].decode("utf-8"))
    assert final_response["choices"][0]["message"]["content"] == " partial final"

    assert [call["api"] for call in select_calls] == [
        proxy.CHAT_COMPLETIONS_API,
        proxy.CHAT_COMPLETIONS_TOKEN_IDS_API,
    ]
    second_select_body = select_calls[1]["req_data"]
    assert "messages" not in second_select_body
    assert second_select_body["prompt_token_ids"] == [101, 102, 201, 202]
    assert second_select_body["max_tokens"] == 7
    assert second_select_body["max_completion_tokens"] == 7

    assert [call["endpoint"] for call in stream_calls] == [
        proxy.CHAT_COMPLETIONS_API,
        proxy.CHAT_COMPLETIONS_TOKEN_IDS_API,
    ]
    second_stream_body = stream_calls[1]["req_data"]
    assert "messages" not in second_stream_body
    assert second_stream_body["prompt_token_ids"] == [101, 102, 201, 202]

    assert proxy.proxy_state.released_prefiller_kv == [(1, 101), (2, 102)]
    assert proxy.proxy_state.released_decoders == [(1, 11), (2, 12)]
    assert proxy.proxy_state.aborted_prefillers == []
