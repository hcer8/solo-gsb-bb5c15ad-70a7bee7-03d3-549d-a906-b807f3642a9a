"""Verification of caller-supplied NOR-flash erase/write sequences.

Where :mod:`.flash_update` *plans* the operations that turn one image
into another, this module *verifies* an externally supplied, ordered
operation list. The operations are replayed one by one against a
simulated NOR flash that starts from the *current* image: writes only
pull bits ``1 -> 0``, erasing restores whole blocks to ``0xFF``, and
addresses never written read as ``0xFF``.

Every operation must satisfy:

* ``erase`` — ``address`` and ``size`` are non-zero multiples of
  ``erase_block_size`` (whole blocks; one op may span several adjacent
  blocks) and the erased range lies inside the flash window;
* ``write`` — the byte range lies inside the flash window and inside a
  single erase block, carries at least one and at most
  ``max_write_size`` bytes, and causes only ``1 -> 0`` bit transitions
  against the replayed state (a ``0 -> 1`` transition requires an erase
  first).

The first offending operation stops the replay and is reported with its
0-based index and an error code. When every operation is legal, the
replayed state is compared with the *target* image (undefined bytes
read as ``0xFF`` on both sides) and the differences are merged into
contiguous ranges; only zero differing bytes means the sequence
faithfully reproduces the target image.
"""

from __future__ import annotations

from dataclasses import dataclass

from .flash_update import ERASE, WRITE, ParamError, validate_flash_params
from .hex_parser import Segment


@dataclass
class ReplayOperation:
    """One caller-supplied erase or write step, replayed in order."""

    kind: str  # erase | write
    address: int
    size: int = 0  # erase: number of bytes to erase
    data: bytes = b""  # write: bytes to program


@dataclass
class OperationError:
    """The first illegal operation; ``index`` is its 0-based position."""

    # erase_unaligned | erase_outside_flash | empty_write |
    # write_outside_flash | write_crosses_block | write_too_large |
    # zero_to_one_write
    kind: str
    index: int
    message: str
    address: int | None = None
    end: int | None = None


@dataclass
class DiffRange:
    """A maximal run of contiguous addresses where replay != target."""

    start: int
    expected: bytes  # target image bytes (0xFF where the image is undefined)
    actual: bytes  # replayed flash bytes (0xFF where undefined)

    @property
    def end(self) -> int:
        """Exclusive end address."""
        return self.start + len(self.expected)


@dataclass
class VerifyResult:
    ok: bool
    params_valid: bool
    param_errors: list[ParamError]
    operations_valid: bool | None  # None when the replay never ran
    operation_error: OperationError | None
    operations_total: int
    operations_checked: int  # operations successfully replayed before stopping
    diff_ranges: list[DiffRange]
    diff_bytes: int


def _fmt(addr: int) -> str:
    return f"0x{addr:08X}"


def _to_map(segments: list[Segment]) -> dict[int, int]:
    """Flatten assembled segments into an ``address -> byte`` mapping."""
    memory: dict[int, int] = {}
    for seg in segments:
        for i, value in enumerate(seg.data):
            memory[seg.start + i] = value
    return memory


def verify_flash_operations(
    current_segments: list[Segment],
    target_segments: list[Segment],
    flash_start: int,
    flash_end: int,
    erase_block_size: int,
    max_write_size: int,
    operations: list[ReplayOperation],
) -> VerifyResult:
    """Replay ``operations`` from the current image and compare with target.

    Flash window/alignment/placement parameters are validated exactly as
    for planning; on any problem the replay does not run
    (``operations_valid`` is None). Otherwise operations are checked and
    applied in order, stopping at the first illegal one. Only when every
    operation is legal is the replayed state diffed against the target
    image; ``ok`` requires zero differing bytes.
    """
    param_errors = validate_flash_params(
        current_segments,
        target_segments,
        flash_start,
        flash_end,
        erase_block_size,
        max_write_size,
    )
    if param_errors:
        return VerifyResult(
            ok=False,
            params_valid=False,
            param_errors=param_errors,
            operations_valid=None,
            operation_error=None,
            operations_total=len(operations),
            operations_checked=0,
            diff_ranges=[],
            diff_bytes=0,
        )

    state = _to_map(current_segments)
    for index, op in enumerate(operations):
        if op.kind == ERASE:
            error = _replay_erase(
                state, op, index, flash_start, flash_end, erase_block_size
            )
        else:
            error = _replay_write(
                state,
                op,
                index,
                flash_start,
                flash_end,
                erase_block_size,
                max_write_size,
            )
        if error is not None:
            return VerifyResult(
                ok=False,
                params_valid=True,
                param_errors=[],
                operations_valid=False,
                operation_error=error,
                operations_total=len(operations),
                operations_checked=index,
                diff_ranges=[],
                diff_bytes=0,
            )

    diff_ranges = _diff_ranges(state, _to_map(target_segments))
    diff_bytes = sum(len(r.expected) for r in diff_ranges)
    return VerifyResult(
        ok=diff_bytes == 0,
        params_valid=True,
        param_errors=[],
        operations_valid=True,
        operation_error=None,
        operations_total=len(operations),
        operations_checked=len(operations),
        diff_ranges=diff_ranges,
        diff_bytes=diff_bytes,
    )


def _replay_erase(
    state: dict[int, int],
    op: ReplayOperation,
    index: int,
    flash_start: int,
    flash_end: int,
    erase_block_size: int,
) -> OperationError | None:
    """Check one erase operation and apply it to the replayed state."""
    size = op.size
    if size <= 0 or op.address % erase_block_size or size % erase_block_size:
        return OperationError(
            kind="erase_unaligned",
            index=index,
            message=(
                f"erase must cover whole {erase_block_size}-byte block(s): address "
                f"{_fmt(op.address)} and size {size} must both be non-zero "
                f"multiple(s) of erase_block_size"
            ),
            address=op.address,
            end=op.address + max(size, 0),
        )
    end = op.address + size
    if op.address < flash_start or end > flash_end:
        return OperationError(
            kind="erase_outside_flash",
            index=index,
            message=(
                f"erase [{_fmt(op.address)}, {_fmt(end)}) lies outside the "
                f"flash window [{_fmt(flash_start)}, {_fmt(flash_end)})"
            ),
            address=op.address,
            end=end,
        )
    # Erasing resets bytes to 0xFF, the value undefined addresses read as.
    for addr in [a for a in state if op.address <= a < end]:
        del state[addr]
    return None


def _replay_write(
    state: dict[int, int],
    op: ReplayOperation,
    index: int,
    flash_start: int,
    flash_end: int,
    erase_block_size: int,
    max_write_size: int,
) -> OperationError | None:
    """Check one write operation and apply it to the replayed state."""
    size = len(op.data)
    end = op.address + size
    if size == 0:
        return OperationError(
            kind="empty_write",
            index=index,
            message=f"write at {_fmt(op.address)} carries no data bytes",
            address=op.address,
            end=end,
        )
    if op.address < flash_start or end > flash_end:
        return OperationError(
            kind="write_outside_flash",
            index=index,
            message=(
                f"write [{_fmt(op.address)}, {_fmt(end)}) lies outside the "
                f"flash window [{_fmt(flash_start)}, {_fmt(flash_end)})"
            ),
            address=op.address,
            end=end,
        )
    if op.address // erase_block_size != (end - 1) // erase_block_size:
        boundary = op.address - (op.address % erase_block_size) + erase_block_size
        return OperationError(
            kind="write_crosses_block",
            index=index,
            message=(
                f"write [{_fmt(op.address)}, {_fmt(end)}) crosses erase-block "
                f"boundary {_fmt(boundary)}"
            ),
            address=op.address,
            end=end,
        )
    if size > max_write_size:
        return OperationError(
            kind="write_too_large",
            index=index,
            message=(
                f"write carries {size} byte(s), exceeding max_write_size "
                f"{max_write_size}"
            ),
            address=op.address,
            end=end,
        )
    for i, value in enumerate(op.data):
        addr = op.address + i
        current = state.get(addr, 0xFF)
        if (~current) & value & 0xFF:
            return OperationError(
                kind="zero_to_one_write",
                index=index,
                message=(
                    f"write at {_fmt(addr)} would flip bit(s) 0 -> 1: replayed "
                    f"state 0x{current:02X} -> 0x{value:02X} requires an erase first"
                ),
                address=addr,
                end=addr + 1,
            )
    for i, value in enumerate(op.data):
        state[op.address + i] = value
    return None


def _diff_ranges(state: dict[int, int], target: dict[int, int]) -> list[DiffRange]:
    """Merge addresses where replay and target differ into contiguous runs.

    Only addresses defined on at least one side are visited; undefined
    bytes read as ``0xFF`` on both sides.
    """
    ranges: list[DiffRange] = []
    run_start: int | None = None
    run_expected = bytearray()
    run_actual = bytearray()
    prev = 0
    for addr in sorted(state.keys() | target.keys()):
        actual = state.get(addr, 0xFF)
        expected = target.get(addr, 0xFF)
        if actual == expected:
            continue
        if run_start is None or addr != prev + 1:
            if run_start is not None:
                ranges.append(
                    DiffRange(run_start, bytes(run_expected), bytes(run_actual))
                )
            run_start = addr
            run_expected = bytearray()
            run_actual = bytearray()
        run_expected.append(expected)
        run_actual.append(actual)
        prev = addr
    if run_start is not None:
        ranges.append(DiffRange(run_start, bytes(run_expected), bytes(run_actual)))
    return ranges
