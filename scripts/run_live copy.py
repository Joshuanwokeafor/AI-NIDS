#!/usr/bin/env python3
"""
scripts/run_live.py
--------------------
Real-time mode: sniffs a live interface, aggregates packets into flow
records, and runs each completed flow through the trained pipeline,
raising alerts as intrusions are detected.

Requires root / CAP_NET_RAW privileges and `scapy` installed.
Note: sniffing requires elevated OS privileges that are unavailable in
this sandbox, so this path is validated by unit/integration tests
against synthetic flow dicts (see tests/) rather than a live NIC here;
the code itself uses only documented scapy APIs.

Usage (Linux, needs sudo):
    sudo python scripts/run_live.py --interface eth0
"""

import argparse
import os

# See scripts/train.py for why this must run before numpy/scipy import.
os.environ.setdefault("OPENBLAS_CORETYPE", "Nehalem")

import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from pipeline import NIDSPipeline  # noqa: E402
from src.feature_extractor import PacketFlowAggregator  # noqa: E402


def run_live(interface: str, flow_timeout: float = 5.0, poll_interval: float = 1.0):
    try:
        from scapy.all import sniff
    except ImportError:
        print("scapy is required for live capture: pip install scapy", file=sys.stderr)
        sys.exit(1)

    pipeline = NIDSPipeline().load()
    aggregator = PacketFlowAggregator(flow_timeout=flow_timeout)
    flow_queue: "queue.Queue[dict]" = queue.Queue()
    stop_event = threading.Event()

    def on_packet(pkt):
        try:
            aggregator.add_packet(pkt)
        except Exception as exc:  # noqa: BLE001 — malformed packet, skip+log
            print(f"[!] Dropped malformed packet: {exc}", file=sys.stderr)

    def sniff_thread():
        sniff(iface=interface, prn=on_packet, store=False,
              stop_filter=lambda _: stop_event.is_set())

    def flow_drain_thread():
        while not stop_event.is_set():
            stop_event.wait(poll_interval)
            for flow in aggregator.pop_expired_flows():
                flow_queue.put(flow)

    t1 = threading.Thread(target=sniff_thread, daemon=True)
    t2 = threading.Thread(target=flow_drain_thread, daemon=True)
    t1.start()
    t2.start()

    print(f"[*] Live capture started on {interface}. Press Ctrl+C to stop.")
    try:
        while True:
            flow = flow_queue.get()
            df = pd.DataFrame([flow])
            alert_ids = pipeline.infer_and_alert(df)
            if alert_ids[0] is not None:
                print(f"[ALERT] id={alert_ids[0]} flow={flow['src_ip']}:{flow['src_port']} "
                      f"-> {flow['dst_ip']}:{flow['dst_port']}")
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[*] Live capture stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run AI-NIDS in real-time live-capture mode.")
    parser.add_argument("--interface", required=True, help="Network interface, e.g. eth0")
    parser.add_argument("--flow-timeout", type=float, default=5.0)
    args = parser.parse_args()
    run_live(args.interface, flow_timeout=args.flow_timeout)


if __name__ == "__main__":
    main()
