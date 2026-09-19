"""
feature_extractor.py
---------------------
Stage 3 of the pipeline: FeatureExtractor.

Ranks the 122 one-hot-expanded features by mutual information I(X;Y)
against the target label and retains the top-k discriminative features
(config.TOP_K_FEATURES). Also contains PacketFlowAggregator, a helper
used by DataIngestor to turn raw packets from a PCAP/live capture into
NSL-KDD-shaped flow records so the rest of the pipeline never needs to
know whether a record originated from CSV or from the wire.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger("ai_nids.feature_extractor")


class FeatureExtractor:
    """Mutual-information based feature selection (Stage 3)."""

    def __init__(self, top_k: int = 40, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: Optional[List[str]] = None
        self.scores_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FeatureExtractor":
        if X.shape[1] == 0:
            raise ValueError("Cannot fit FeatureExtractor on zero features.")
        k = min(self.top_k, X.shape[1])
        mi_scores = mutual_info_classif(
            X, y, discrete_features=False, random_state=self.random_state
        )
        self.scores_ = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        self.selected_features_ = list(self.scores_.index[:k])
        logger.info("FeatureExtractor selected %d/%d features (top_k=%d)",
                    len(self.selected_features_), X.shape[1], self.top_k)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("FeatureExtractor.fit() must be called before transform().")
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing selected features: {missing}")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    def top_features_report(self, n: int = 20) -> pd.Series:
        if self.scores_ is None:
            raise RuntimeError("Call fit() first.")
        return self.scores_.head(n)


class PacketFlowAggregator:
    """
    Groups raw packets (from scapy) into bidirectional flows keyed by the
    5-tuple (src_ip, dst_ip, src_port, dst_port, protocol) and computes a
    reduced set of NSL-KDD-style statistical features per flow:
    duration, src_bytes, dst_bytes, count, srv_count, protocol_type, flag.

    This is intentionally a simplified, dependency-light flow featurizer
    (a full CICFlowMeter-equivalent is out of scope for the artifact) but
    is sufficient to feed the same Preprocessor / FeatureExtractor /
    classifier pipeline used for the NSL-KDD CSV path, satisfying FR1
    (multi-mode ingestion) end-to-end.
    """

    def __init__(self, flow_timeout: float = 5.0):
        self.flow_timeout = flow_timeout
        self._flows: Dict[tuple, dict] = {}
        self._service_counts: Dict[str, int] = defaultdict(int)

    def _flow_key(self, pkt) -> Optional[tuple]:
        if not pkt.haslayer("IP"):
            return None
        ip = pkt["IP"]
        proto = ip.proto  # 6=TCP, 17=UDP, 1=ICMP
        sport = dport = 0
        if pkt.haslayer("TCP"):
            sport, dport = pkt["TCP"].sport, pkt["TCP"].dport
        elif pkt.haslayer("UDP"):
            sport, dport = pkt["UDP"].sport, pkt["UDP"].dport
        return (ip.src, ip.dst, sport, dport, proto)

    def add_packet(self, pkt) -> None:
        key = self._flow_key(pkt)
        if key is None:
            raise ValueError("Non-IP packet, cannot assign to a flow")

        now = float(getattr(pkt, "time", time.time()))
        size = len(bytes(pkt))

        flow = self._flows.get(key)
        if flow is None:
            proto_name = {6: "tcp", 17: "udp", 1: "icmp"}.get(key[4], "other")
            flow = {
                "start": now, "last": now,
                "src_ip": key[0], "dst_ip": key[1],
                "src_port": key[2], "dst_port": key[3],
                "protocol_type": proto_name,
                "src_bytes": 0, "dst_bytes": 0,
                "packet_count": 0,
                "flag": "SF",
            }
            self._flows[key] = flow

        flow["last"] = now
        flow["packet_count"] += 1
        # crude directionality: treat the initiating IP as "src"
        if key[0] == flow["src_ip"]:
            flow["src_bytes"] += size
        else:
            flow["dst_bytes"] += size

        if pkt.haslayer("TCP"):
            flags = pkt["TCP"].flags
            if "R" in str(flags):
                flow["flag"] = "REJ"
            elif "S" in str(flags) and "A" not in str(flags):
                flow["flag"] = "S0"

    def pop_expired_flows(self) -> List[dict]:
        now = time.time()
        expired_keys = [
            k for k, f in self._flows.items() if now - f["last"] > self.flow_timeout
        ]
        results = [self._flow_to_record(self._flows.pop(k)) for k in expired_keys]
        return results

    def finalize_flows(self) -> List[dict]:
        results = [self._flow_to_record(f) for f in self._flows.values()]
        self._flows.clear()
        return results

    def _flow_to_record(self, flow: dict) -> dict:
        duration = max(flow["last"] - flow["start"], 0.0)
        service = self._guess_service(flow["dst_port"])
        return {
            "duration": duration,
            "protocol_type": flow["protocol_type"],
            "service": service,
            "flag": flow["flag"],
            "src_bytes": flow["src_bytes"],
            "dst_bytes": flow["dst_bytes"],
            "count": flow["packet_count"],
            "srv_count": flow["packet_count"],
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
        }

    @staticmethod
    def _guess_service(port: int) -> str:
        well_known = {
            80: "http", 443: "https", 21: "ftp", 22: "ssh", 23: "telnet",
            25: "smtp", 53: "domain", 110: "pop3", 143: "imap4",
            3306: "mysql", 3389: "ms_wbt", 445: "microsoft_ds",
        }
        return well_known.get(port, "other")
