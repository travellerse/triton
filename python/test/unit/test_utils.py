from types import SimpleNamespace

import pytest

import triton
from triton import _internal_testing
from triton._utils import is_power_of_two, validate_block_shape
from triton.runtime import driver
from triton.testing import get_dram_gbps


@pytest.mark.parametrize("y", [1, 32])
def test_cdiv_symbolic_numerator(y):

    class Expr:

        def __init__(self, evaluate):
            self.evaluate = evaluate

        def __add__(self, other):
            return Expr(lambda x: self.evaluate(x) + other)

        def __floordiv__(self, other):
            return Expr(lambda x: self.evaluate(x) // other)

    result = triton.cdiv(Expr(lambda x: x), y)
    assert isinstance(result, Expr)
    for x in [0, 1, 31, 32, 33, 64, 65]:
        assert result.evaluate(x) == -(-x // y)


@pytest.mark.parametrize("x", [-33, -1, 0, 1, 31, 32, 33, 2**100, 2**100 + 1])
@pytest.mark.parametrize("y", [1, 2, 32, 2**64])
def test_cdiv_integers(x, y):
    assert triton.cdiv(x, y) == -(-x // y)


def test_cdiv_zero_divisor():
    with pytest.raises(ZeroDivisionError):
        triton.cdiv(1, 0)


def test_is_power_of_two():
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(8)
    assert is_power_of_two(1024)
    # 0 is not a power of two; x & (x - 1) == 0 alone wrongly accepts it.
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(6)
    assert not is_power_of_two(-4)


def test_validate_block_shape_rejects_zero():
    # validate_block_shape promises every element is a power of 2, but a 0
    # element used to slip through because is_power_of_two(0) returned True.
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([0])
    with pytest.raises(ValueError, match="must be a power of 2"):
        validate_block_shape([8, 0])


def test_validate_block_shape_accepts_powers_of_two():
    assert validate_block_shape([8, 16]) == 128


@pytest.mark.parametrize(
    "backend, arch, properties, expected",
    [
        ("hip", "gfx1100", {"mem_clock_rate": 1124000, "mem_bus_width": 384}, 863.232),
        ("hip", "gfx1101", {"mem_clock_rate": 1218750, "mem_bus_width": 256}, 624.0),
        ("hip", "gfx1102", {"mem_clock_rate": 1125000, "mem_bus_width": 128}, 288.0),
        ("hip", "gfx1151", {"mem_clock_rate": 1000000, "mem_bus_width": 256}, 256.0),
        ("hip", "gfx1151", {"mem_clock_rate": 1000000, "mem_bus_width": 128}, 128.0),
        ("hip", "gfx1200", {"mem_clock_rate": 1258000, "mem_bus_width": 128}, 322.048),
        ("hip", "gfx1201", {"mem_clock_rate": 1250000, "mem_bus_width": 256}, 640.0),
        ("cuda", 90, {"mem_clock_rate": 1000000, "mem_bus_width": 256}, 64.0),
    ],
)
def test_get_dram_gbps_memory_data_rate(monkeypatch, backend, arch, properties, expected):
    utils = SimpleNamespace(get_device_properties=lambda _device: properties)
    target = SimpleNamespace(backend=backend, arch=arch)
    active = SimpleNamespace(utils=utils, get_current_target=lambda: target)
    monkeypatch.setattr(driver, "_active", active)
    assert get_dram_gbps(0) == expected


@pytest.mark.parametrize("backend, arch, expected", [
    ("hip", "gfx1030", True),
    ("hip", "gfx1031", True),
    ("hip", "gfx1036", True),
    ("hip", "gfx1012", False),   # RDNA1 is not supported by this backend
    ("hip", "gfx1100", False),
    ("hip", "gfx1151", False),
    ("hip", "gfx1201", False),
    ("hip", "gfx1250", False),
    ("hip", "gfx90a", False),
    ("hip", "gfx942", False),
    ("hip", "gfx950", False),
    ("cuda", "gfx1030", False),
])
def test_is_hip_rdna2_recognizes_gfx103_only(monkeypatch, backend, arch, expected):
    # RDNA2 is the GFX10.3 stepping range, which is exactly what computeISAFamily() maps to
    # ISAFamily::RDNA2 (major == 10 && minor == 3).
    monkeypatch.setattr(_internal_testing, "get_current_target",
                        lambda: SimpleNamespace(backend=backend, arch=arch))
    assert _internal_testing.is_hip_rdna2() is expected


def test_is_hip_rdna2_does_not_imply_other_family_gates(monkeypatch):
    # is_hip_rdna() and the CDNA predicates drive RDNA3/RDNA4- and CDNA-only expectations
    # (WMMA layout versions, warp specialization, the triton-to-gluon translator skip).
    # Recognising RDNA2 must not move a gfx103x device into any of those groups.
    monkeypatch.setattr(_internal_testing, "get_current_target",
                        lambda: SimpleNamespace(backend="hip", arch="gfx1031"))
    assert _internal_testing.is_hip()
    assert _internal_testing.is_hip_rdna2()
    assert not _internal_testing.is_hip_rdna()
    assert not _internal_testing.is_hip_rdna3()
    assert not _internal_testing.is_hip_cdna()
    assert not _internal_testing.is_hip_cdna3_or_newer()
