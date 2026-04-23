import asyncio
import importlib.util
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "disaggregated_prefill_v1"
    / "load_balance_proxy_server_example.py"
)


def load_proxy_module():
    spec = importlib.util.spec_from_file_location("load_balance_proxy_server_example", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_recompute_instance_releases_previous_decoder():
    module = load_proxy_module()

    class FakeScheduler:
        def __init__(self):
            self.calls = []

        def release_decoder(self, decoder_key, decoder_score):
            self.calls.append((decoder_key, decoder_score))

    scheduler = FakeScheduler()
    module.runtime = SimpleNamespace(scheduler=scheduler)

    expected = module.InstanceInfo(
        request_id="new",
        prefiller_key="prefill-1",
        prefiller_score=1.0,
        prefiller={"key": "prefill-1", "host": "127.0.0.1", "port": 8100},
        decoder_key="decode-2",
        decoder_score=2.0,
        decoder={"key": "decode-2", "host": "127.0.0.1", "port": 8201},
    )

    async def fake_handle_select_instance(api, req_data, request_length):
        assert api == "/completions"
        assert req_data["prompt"] == "retry"
        assert request_length == 42
        return expected

    previous = module.InstanceInfo(
        request_id="old",
        prefiller_key="prefill-0",
        prefiller_score=1.0,
        prefiller={"key": "prefill-0", "host": "127.0.0.1", "port": 8100},
        decoder_key="decode-1",
        decoder_score=9.5,
        decoder={"key": "decode-1", "host": "127.0.0.1", "port": 8200},
    )

    with mock.patch.object(module, "handle_select_instance", fake_handle_select_instance):
        result = asyncio.run(
            module.select_recompute_instance("/completions", {"prompt": "retry"}, 42, previous)
        )

    assert result == expected
    assert scheduler.calls == [("decode-1", 9.5)]


def test_shared_scheduler_survives_100000_concurrent_requests():
    module = load_proxy_module()
    scheduler = module.SharedProxyScheduler(
        prefiller_instances=[("127.0.0.1", 8100), ("127.0.0.1", 8101), ("127.0.0.1", 8102), ("127.0.0.1", 8103)],
        decoder_instances=[("127.0.0.1", 8200), ("127.0.0.1", 8201), ("127.0.0.1", 8202), ("127.0.0.1", 8203)],
    )

    total_requests = 100000

    def one_request(idx: int) -> None:
        request_length = 128 + (idx % 97)
        prefill_score = scheduler.calculate_prefill_scores(request_length)
        decode_score = scheduler.calculate_decode_scores(request_length)
        scheduler.request_started()
        prefiller = scheduler.select_prefiller(prefill_score)
        decoder = scheduler.select_decoder(decode_score)
        scheduler.release_prefiller(prefiller["key"], prefill_score)
        scheduler.release_prefiller_kv(prefiller["key"], prefill_score)
        scheduler.release_decoder(decoder["key"], decode_score)
        scheduler.request_finished()

    with ThreadPoolExecutor(max_workers=256) as executor:
        for _ in executor.map(one_request, range(total_requests), chunksize=128):
            pass

    assert scheduler.request_num == 0
    assert len(scheduler.prefiller_heap) == len(scheduler.prefillers)
    assert len(scheduler.decoder_heap) == len(scheduler.decoders)

    for server in scheduler.prefillers.values():
        assert math.isclose(server["active_tokens"], 0.0, abs_tol=1e-9)
        assert math.isclose(server["active_kv_cache"], 0.0, abs_tol=1e-9)

    for server in scheduler.decoders.values():
        assert math.isclose(server["active_tokens"], 0.0, abs_tol=1e-9)
