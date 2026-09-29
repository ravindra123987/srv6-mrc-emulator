"""Tests for srv6_mrc.rdma (RoCEv2 BTH payload framing) and the pure
`topo.dqpn_for_ev` / `topo.ev_from_dqpn` addressing helpers it uses.

Split into two groups:
  - Pure-Python (topo.py) tests always run — no scapy needed.
  - BTH wrap/unwrap round-trip tests need scapy; skipped on a dev
    laptop without it (mirrors tests/test_mrc_transport.py's
    _HAVE_SCAPY pattern) but run inside the alpine host image / any
    box with `pip install scapy`.
"""
from __future__ import annotations

import unittest

from srv6_mrc import topo
from srv6_mrc.runner import encode_payload, parse_payload

try:
    import scapy.all  # type: ignore # noqa: F401
    _HAVE_SCAPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCAPY = False


class TestDqpnForEv(unittest.TestCase):
    def test_round_trip(self):
        for plane in range(topo.NUM_PLANES):
            for path in range(topo.NUM_SPINES):
                dqpn = topo.dqpn_for_ev(plane, path)
                self.assertEqual(topo.ev_from_dqpn(dqpn), (plane, path))

    def test_known_shape(self):
        # plane=2, path=5 -> (2 << 8) | 5 = 0x205
        self.assertEqual(topo.dqpn_for_ev(2, 5), 0x205)

    def test_bad_plane_rejected(self):
        with self.assertRaises(ValueError):
            topo.dqpn_for_ev(topo.NUM_PLANES, 0)

    def test_bad_path_rejected(self):
        with self.assertRaises(ValueError):
            topo.dqpn_for_ev(0, topo.NUM_SPINES)


class TestTransportValidation(unittest.TestCase):
    def test_transports_tuple(self):
        self.assertEqual(topo.TRANSPORTS, ("udp", "rdma"))

    def test_check_transport_accepts_valid(self):
        topo._check_transport("udp")
        topo._check_transport("rdma")

    def test_check_transport_rejects_bad(self):
        with self.assertRaises(ValueError):
            topo._check_transport("tcp")

    def test_rdma_port_is_iana_roce_port(self):
        self.assertEqual(topo.RDMA_PORT, 4791)


@unittest.skipUnless(_HAVE_SCAPY, "scapy not installed")
class TestWrapUnwrapRdma(unittest.TestCase):
    def test_round_trip_recovers_mrc_payload(self):
        from srv6_mrc.rdma import wrap_rdma, unwrap_rdma

        payload = encode_payload(seq=123456, plane=2, path=5)
        wrapped = wrap_rdma(payload, plane=2, path=5, seq=123456)
        unwrapped = unwrap_rdma(wrapped)
        self.assertEqual(unwrapped, payload)
        self.assertEqual(parse_payload(unwrapped), (123456, 2, 5))

    def test_wrapped_is_longer_by_bth_header_and_icrc(self):
        from srv6_mrc.rdma import wrap_rdma

        payload = encode_payload(seq=1, plane=0, path=0)
        wrapped = wrap_rdma(payload, plane=0, path=0, seq=1)
        # BTH fixed header (12B) + payload + ICRC trailer (4B).
        self.assertEqual(len(wrapped), len(payload) + 12 + 4)

    def test_wrapped_carries_dport_binding_opcode_and_dqpn(self):
        from srv6_mrc.rdma import wrap_rdma, RDMA_OPCODE
        from scapy.contrib.roce import BTH

        payload = encode_payload(seq=99, plane=1, path=3)
        wrapped = wrap_rdma(payload, plane=1, path=3, seq=99)
        pkt = BTH(wrapped)
        self.assertEqual(pkt.opcode, RDMA_OPCODE)
        self.assertEqual(pkt.dqpn, topo.dqpn_for_ev(1, 3))
        self.assertEqual(pkt.psn, 99)

    def test_psn_truncates_to_24_bits(self):
        from srv6_mrc.rdma import wrap_rdma
        from scapy.contrib.roce import BTH

        big_seq = (1 << 40) + 42  # far beyond a 24-bit PSN
        payload = encode_payload(seq=big_seq & 0xFFFFFFFFFFFFFFFF,
                                  plane=0, path=0)
        wrapped = wrap_rdma(payload, plane=0, path=0, seq=big_seq)
        pkt = BTH(wrapped)
        self.assertEqual(pkt.psn, big_seq & 0xFFFFFF)

    def test_unwrap_rejects_too_short(self):
        from srv6_mrc.rdma import unwrap_rdma

        self.assertIsNone(unwrap_rdma(b"\x00" * 4))

    def test_unwrap_rejects_empty(self):
        from srv6_mrc.rdma import unwrap_rdma

        self.assertIsNone(unwrap_rdma(b""))


if __name__ == "__main__":
    unittest.main()
