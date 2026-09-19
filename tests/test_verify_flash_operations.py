"""Tests for the /api/v1/verify-flash-operations endpoint and replay logic."""

import base64

from fastapi.testclient import TestClient

from app.hex_parser import Segment
from app.main import app
from app.flash_verify import ReplayOperation, verify_flash_operations

client = TestClient(app)


def rec(rectype: int, addr: int, data: bytes = b"") -> str:
    """Build a valid Intel HEX record line."""
    body = bytes([len(data), (addr >> 8) & 0xFF, addr & 0xFF, rectype]) + bytes(data)
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes([checksum])).hex().upper()


EOF = rec(0x01, 0)


def data_rec(addr: int, data: bytes) -> str:
    return rec(0x00, addr, data)


def hex_text(*records: str) -> str:
    return "\n".join([*records, EOF])


def seg(start: int, data: bytes) -> Segment:
    return Segment(start=start, data=data)


def erase(address: int, size: int) -> ReplayOperation:
    return ReplayOperation(kind="erase", address=address, size=size)


def write(address: int, data: bytes) -> ReplayOperation:
    return ReplayOperation(kind="write", address=address, data=data)


def verify(current, target, ops, start=0, end=0x100, block=0x40, max_write=0x10):
    return verify_flash_operations(current, target, start, end, block, max_write, ops)


# ---------------------------------------------------------------------------
# pure replay/verify logic
# ---------------------------------------------------------------------------

def test_no_operations_identical_images_ok():
    result = verify([seg(0, bytes(range(16)))], [seg(0, bytes(range(16)))], [])
    assert result.ok
    assert result.params_valid
    assert result.operations_valid is True
    assert result.operation_error is None
    assert result.operations_total == 0
    assert result.operations_checked == 0
    assert result.diff_ranges == []
    assert result.diff_bytes == 0


def test_no_operations_differing_images_report_merged_diff():
    # differ at 0x00..0x01 (contiguous) and 0x03 (separate run)
    current = [seg(0x00, b"\x00\x00\xAA\x00")]
    target = [seg(0x00, b"\x11\x22\xAA\x33")]
    result = verify(current, target, [])
    assert not result.ok
    assert result.operations_valid is True
    assert result.operations_checked == 0
    assert [(r.start, r.end) for r in result.diff_ranges] == [(0x00, 0x02), (0x03, 0x04)]
    assert result.diff_ranges[0].expected == b"\x11\x22"
    assert result.diff_ranges[0].actual == b"\x00\x00"
    assert result.diff_bytes == 3


def test_legal_erase_write_sequence_reaches_target():
    # 0x00 -> 0xFF needs an erase; the block is then rewritten
    current = [seg(0x00, bytes(range(16)))]
    target = [seg(0x00, b"\xFF" + bytes(range(1, 16)))]
    ops = [erase(0x00, 0x40), write(0x01, bytes(range(1, 16)))]
    result = verify(current, target, ops)
    assert result.ok
    assert result.operations_valid is True
    assert result.operations_checked == 2
    assert result.diff_bytes == 0


def test_erase_spanning_multiple_blocks_accepted():
    current = [seg(0x00, b"\x00" * 8), seg(0x40, b"\x00" * 8)]
    target = [seg(0x00, b"\xFF" * 8), seg(0x40, b"\xFF" * 8)]
    result = verify(current, target, [erase(0x00, 0x80)])
    assert result.ok
    assert result.diff_bytes == 0


def test_erase_unaligned_address_rejected():
    result = verify([], [], [erase(0x08, 0x40)])
    assert not result.ok
    assert result.operations_valid is False
    error = result.operation_error
    assert error.kind == "erase_unaligned"
    assert error.index == 0
    assert result.operations_checked == 0


def test_erase_size_not_multiple_of_block_rejected():
    result = verify([], [], [erase(0x00, 0x30)])
    assert result.operation_error.kind == "erase_unaligned"
    # zero-size erase is not a whole block either
    result = verify([], [], [erase(0x00, 0)])
    assert result.operation_error.kind == "erase_unaligned"


def test_erase_outside_flash_rejected():
    result = verify([], [], [erase(0x00, 0x40)], start=0x40, end=0x80)
    error = result.operation_error
    assert error.kind == "erase_outside_flash"
    assert (error.address, error.end) == (0x00, 0x40)


def test_write_outside_flash_rejected():
    # starts exactly at flash_end; a write ending at flash_end would be legal
    result = verify([], [], [write(0x100, b"\x01")], end=0x100)
    assert result.operation_error.kind == "write_outside_flash"
    result = verify([], [], [write(0xFC, b"\x01" * 8)], end=0x100)
    assert result.operation_error.kind == "write_outside_flash"


def test_write_crossing_block_boundary_rejected():
    # 0x3E..0x42 straddles the 0x40 block boundary
    result = verify([], [], [write(0x3E, b"\x00" * 4)], max_write=0x10)
    error = result.operation_error
    assert error.kind == "write_crosses_block"
    assert (error.address, error.end) == (0x3E, 0x42)


def test_write_larger_than_max_write_size_rejected():
    result = verify([], [], [write(0x00, b"\x00" * 8)], max_write=4)
    assert result.operation_error.kind == "write_too_large"


def test_empty_write_rejected():
    result = verify([], [], [write(0x00, b"")])
    assert result.operation_error.kind == "empty_write"


def test_zero_to_one_write_rejected():
    current = [seg(0x00, b"\x0F")]
    result = verify(current, [], [write(0x00, b"\xFF")])
    error = result.operation_error
    assert error.kind == "zero_to_one_write"
    assert error.address == 0x00
    assert result.operations_checked == 0


def test_first_illegal_operation_stops_replay():
    # op 0 is a legal write; op 1 needs bits the (un-erased) state no longer has
    current = [seg(0x00, b"\xFF")]
    ops = [write(0x00, b"\x0F"), write(0x00, b"\xF0"), write(0x00, b"\x00")]
    result = verify(current, [], ops)
    assert result.operations_valid is False
    error = result.operation_error
    assert error.index == 1
    assert error.kind == "zero_to_one_write"
    assert result.operations_checked == 1
    # the illegal write and everything after it were not applied: no diff ran
    assert result.diff_ranges == []


def test_replay_state_accumulates_across_writes():
    # two legal 1->0 writes to the same byte compose: 0xFF -> 0x0F -> 0x07
    current = [seg(0x00, b"\xFF")]
    target = [seg(0x00, b"\x07")]
    ops = [write(0x00, b"\x0F"), write(0x00, b"\x07")]
    result = verify(current, target, ops)
    assert result.ok
    assert result.operations_checked == 2


def test_erase_restores_ff_so_zero_to_one_becomes_legal():
    current = [seg(0x00, b"\x00")]
    target = [seg(0x00, b"\xFF")]
    result = verify(current, target, [erase(0x00, 0x40)])
    assert result.ok
    # without the erase the same transition is illegal
    result = verify(current, target, [write(0x00, b"\xFF")])
    assert result.operation_error.kind == "zero_to_one_write"


def test_undefined_bytes_compare_as_ff():
    # current explicitly defines 0xFF at 0x00; target defines nothing:
    # both sides read 0xFF -> no diff
    result = verify([seg(0x00, b"\xFF")], [], [])
    assert result.ok
    assert result.diff_bytes == 0


def test_diff_covers_bytes_defined_on_either_side():
    # replay leaves 0x00 at 0x10 (target: implicit 0xFF) and misses the
    # target byte at 0x20 (replayed: implicit 0xFF)
    current = [seg(0x10, b"\x00")]
    target = [seg(0x20, b"\x01")]
    result = verify(current, target, [])
    assert not result.ok
    assert [(r.start, r.expected, r.actual) for r in result.diff_ranges] == [
        (0x10, b"\xFF", b"\x00"),
        (0x20, b"\x01", b"\xFF"),
    ]
    assert result.diff_bytes == 2


def test_params_invalid_skips_replay():
    result = verify([], [], [erase(0x00, 0x40)], start=0x01, end=0x40)
    assert not result.ok
    assert result.params_valid is False
    assert [e.kind for e in result.param_errors] == ["erase_block_unaligned"]
    assert result.operations_valid is None
    assert result.operation_error is None
    assert result.operations_checked == 0
    assert result.diff_ranges == []


def test_image_outside_flash_skips_replay():
    result = verify([seg(0x00, b"\x01")], [], [], start=0x40, end=0x80)
    assert result.params_valid is False
    assert [e.kind for e in result.param_errors] == ["image_outside_flash"]
    assert result.operations_valid is None


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def api_payload(current, target, operations, **overrides):
    payload = {
        "current_hex": current,
        "target_hex": target,
        "flash_start": 0,
        "flash_end": 0x100,
        "erase_block_size": 0x40,
        "max_write_size": 0x10,
        "operations": operations,
    }
    payload.update(overrides)
    return payload


def test_api_happy_path_ok():
    current = hex_text(data_rec(0x0000, bytes(range(16))))
    target = hex_text(data_rec(0x0000, b"\xFF" + bytes(range(1, 16))))
    resp = client.post(
        "/api/v1/verify-flash-operations",
        json=api_payload(
            current,
            target,
            [
                {"kind": "erase", "address": 0, "size": 64},
                {"kind": "write", "address": 1, "data_base64": base64.b64encode(bytes(range(1, 16))).decode()},
            ],
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["current_hex_ok"] is True
    assert body["target_hex_ok"] is True
    assert body["params_valid"] is True
    assert body["operations_valid"] is True
    assert body["operations_total"] == 2
    assert body["operations_checked"] == 2
    assert body["operation_error"] is None
    assert body["diff_ranges"] == []
    assert body["diff_bytes"] == 0
    assert body["current_diagnostics"]
    assert body["target_diagnostics"]


def test_api_operation_error_reported_with_index_and_kind():
    current = hex_text(data_rec(0x0000, b"\x00"))
    target = hex_text(data_rec(0x0000, b"\xFF"))
    resp = client.post(
        "/api/v1/verify-flash-operations",
        json=api_payload(
            current,
            target,
            [
                {"kind": "write", "address": 5, "data_base64": base64.b64encode(b"\x01").decode()},
                {"kind": "write", "address": 0, "data_base64": base64.b64encode(b"\xFF").decode()},
            ],
        ),
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["operations_valid"] is False
    error = body["operation_error"]
    assert error["index"] == 1
    assert error["kind"] == "zero_to_one_write"
    assert error["address"] == 0
    assert body["operations_checked"] == 1
    assert body["diff_ranges"] == []


def test_api_legal_operations_with_diff_return_ok_false():
    # legal write, but the replayed flash does not match the target
    current = hex_text(data_rec(0x0000, b"\xFF\xFF"))
    target = hex_text(data_rec(0x0000, b"\x12\x34"))
    resp = client.post(
        "/api/v1/verify-flash-operations",
        json=api_payload(
            current,
            target,
            [{"kind": "write", "address": 0, "data_base64": base64.b64encode(b"\x12").decode()}],
        ),
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["operations_valid"] is True
    assert body["operation_error"] is None
    assert body["diff_ranges"] == [
        {
            "start": 1,
            "end": 2,
            "size": 1,
            "expected_base64": base64.b64encode(b"\x34").decode(),
            "actual_base64": base64.b64encode(b"\xFF").decode(),
        }
    ]
    assert body["diff_bytes"] == 1


def test_api_invalid_hex_skips_replay():
    good = hex_text(data_rec(0x0000, b"\x01"))
    resp = client.post(
        "/api/v1/verify-flash-operations",
        json=api_payload(
            ":00000001FE\n",  # bad checksum
            good,
            [{"kind": "erase", "address": 0, "size": 64}],
        ),
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["current_hex_ok"] is False
    assert body["target_hex_ok"] is True
    assert body["params_valid"] is None
    assert body["operations_valid"] is None
    assert body["operation_error"] is None
    assert body["operations_checked"] == 0
    assert body["operations_total"] == 1
    assert body["diff_ranges"] == []
    assert any("checksum" in d["message"] for d in body["current_diagnostics"])


def test_api_param_error_skips_replay():
    hex_text_ = hex_text(data_rec(0x0000, b"\x01"))
    resp = client.post(
        "/api/v1/verify-flash-operations",
        json=api_payload(
            hex_text_,
            hex_text_,
            [{"kind": "erase", "address": 0, "size": 64}],
            flash_start=1,
            flash_end=65,
        ),
    )
    body = resp.json()
    assert body["ok"] is False
    assert body["params_valid"] is False
    assert [e["kind"] for e in body["param_errors"]] == ["erase_block_unaligned"]
    assert body["operations_valid"] is None
    assert body["operation_error"] is None


def test_api_malformed_operations_are_422():
    good = hex_text(data_rec(0x0000, b"\x01"))
    bad_ops = [
        {"kind": "erase", "address": 0},  # erase without size
        {"kind": "erase", "address": 0, "size": 64, "data_base64": "AA=="},
        {"kind": "write", "address": 0},  # write without data_base64
        {"kind": "write", "address": 0, "data_base64": "!!!"},
        {"kind": "write", "address": 0, "size": 2, "data_base64": "AA=="},
        {"kind": "read", "address": 0},
    ]
    for op in bad_ops:
        resp = client.post(
            "/api/v1/verify-flash-operations",
            json=api_payload(good, good, [op]),
        )
        assert resp.status_code == 422, op


def test_api_plan_flash_update_output_verifies_ok():
    # the planner's own operation list must verify cleanly: extra fields
    # (end/block) are ignored and write sizes match their data
    current = hex_text(data_rec(0x0000, b"\x00\x01\x02\x03"))
    target = hex_text(data_rec(0x0000, b"\xFF\x01\x03\x03"))
    plan = client.post(
        "/api/v1/plan-flash-update",
        json={
            "current_hex": current,
            "target_hex": target,
            "flash_start": 0,
            "flash_end": 0x100,
            "erase_block_size": 0x40,
            "max_write_size": 0x10,
        },
    ).json()
    assert plan["ok"] is True
    resp = client.post(
        "/api/v1/verify-flash-operations",
        json=api_payload(current, target, plan["operations"]),
    )
    body = resp.json()
    assert body["operations_valid"] is True
    assert body["ok"] is True
    assert body["diff_bytes"] == 0
