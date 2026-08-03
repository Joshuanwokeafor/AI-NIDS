"""
data_ingestor.py
-----------------
Stage 1 of the pipeline: DataIngestor.

Responsible for loading traffic data from two sources:
    * NSL-KDD formatted CSV files (batch / training / offline evaluation)
    * Raw PCAP capture, either a saved .pcap file or a live network
      interface (real-time mode), using scapy.

Malformed records are rejected and logged rather than raising and
killing the whole run, per the functional requirement that ingestion
must be resilient to dirty input.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import pandas as pd

from config import NSL_KDD_COLUMNS, LOG_DIR

logger = logging.getLogger("ai_nids.data_ingestor")
if not logger.handlers:
    handler = logging.FileHandler(os.path.join(LOG_DIR, "ingestion.log"))
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class IngestionError(Exception):
    """Raised when a data source cannot be opened at all."""


@dataclass
class IngestionReport:
    """Summary returned alongside ingested data so callers/tests can
    verify how many records were accepted vs. rejected."""
    total_seen: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejection_reasons": self.rejection_reasons[:50],  # cap for sanity
        }


class DataIngestor:
    """
    Handles Stage 1 (Data Ingestion) of the AI-NIDS pipeline.

    Two entry points:
        load_csv(path)          -> (DataFrame, IngestionReport)
        load_pcap(path)         -> (DataFrame, IngestionReport)   [offline]
        stream_live(interface)  -> generator of feature dicts       [online]
    """

    #: Numeric columns that MUST parse as floats/ints for a record to be valid
    REQUIRED_NUMERIC_COLUMNS = [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
    ]

    def __init__(self, expected_columns: Optional[List[str]] = None):
        self.expected_columns = expected_columns or NSL_KDD_COLUMNS

    # ------------------------------------------------------------------
    # CSV ingestion (NSL-KDD)
    # ------------------------------------------------------------------
    def load_csv(self, path: str) -> "tuple[pd.DataFrame, IngestionReport]":
        if not os.path.exists(path):
            raise IngestionError(f"CSV file not found: {path}")

        report = IngestionReport()
        valid_rows = []

        with open(path, newline="") as f:
            reader = csv.reader(f)

            first_row = next(reader, None)
            if first_row is None:
                raise IngestionError(f"CSV file is empty: {path}")

            # Header detection: NSL-KDD files may or may not ship a header
            # row. Rather than rely on csv.Sniffer (which fails to guess a
            # delimiter on small/numeric-heavy samples), check directly
            # whether the first row contains recognizable schema column
            # names — a much more reliable signal for this fixed schema.
            first_row_stripped = [c.strip() for c in first_row]
            looks_like_header = any(
                col in first_row_stripped for col in ("protocol_type", "class", "duration")
            )

            if looks_like_header:
                header = first_row_stripped
            else:
                header = self.expected_columns[: len(first_row)]
                # first_row was actually data, not a header -> validate it too
                report.total_seen += 1
                row_dict = self._validate_row(first_row, header, report)
                if row_dict is not None:
                    valid_rows.append(row_dict)

            for raw_row in reader:
                report.total_seen += 1
                row_dict = self._validate_row(raw_row, header, report)
                if row_dict is not None:
                    valid_rows.append(row_dict)

        df = pd.DataFrame(valid_rows)
        # Normalize column set to the canonical NSL-KDD schema where possible
        df = self._coerce_dtypes(df)
        logger.info(
            "load_csv(%s): accepted=%d rejected=%d",
            path, report.accepted, report.rejected,
        )
        return df, report

    def _validate_row(self, raw_row: List[str], header: List[str],
                       report: IngestionReport) -> Optional[dict]:
        if len(raw_row) != len(header):
            report.rejected += 1
            report.rejection_reasons.append(
                f"column count mismatch: expected {len(header)}, got {len(raw_row)}"
            )
            return None

        row_dict = dict(zip(header, raw_row))

        for col in self.REQUIRED_NUMERIC_COLUMNS:
            if col not in row_dict:
                continue
            try:
                float(row_dict[col])
            except (ValueError, TypeError):
                report.rejected += 1
                report.rejection_reasons.append(
                    f"non-numeric value in required column '{col}': {row_dict.get(col)!r}"
                )
                return None

        report.accepted += 1
        return row_dict

    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        for col in df.columns:
            if col not in ("protocol_type", "service", "flag", "class"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # difficulty column (NSL-KDD specific) is optional metadata, not a feature
        return df

    # ------------------------------------------------------------------
    # PCAP ingestion (offline file)
    # ------------------------------------------------------------------
    def load_pcap(self, path: str) -> "tuple[pd.DataFrame, IngestionReport]":
        """
        Parses a saved .pcap/.pcapng file into per-flow feature dicts using
        scapy. Requires `scapy` to be installed; raises IngestionError with
        a clear message if it is not available (keeps the CSV/training path
        usable in environments without packet-capture libraries).
        """
        if not os.path.exists(path):
            raise IngestionError(f"PCAP file not found: {path}")

        try:
            from scapy.all import rdpcap  # local import: optional dependency
        except ImportError as exc:
            raise IngestionError(
                "scapy is required for PCAP ingestion. Install with "
                "'pip install scapy'."
            ) from exc

        report = IngestionReport()
        rows = []
        try:
            packets = rdpcap(path)
        except Exception as exc:  # malformed capture file
            raise IngestionError(f"Failed to parse PCAP file {path}: {exc}") from exc

        from .feature_extractor import PacketFlowAggregator
        aggregator = PacketFlowAggregator()

        for pkt in packets:
            report.total_seen += 1
            try:
                aggregator.add_packet(pkt)
                report.accepted += 1
            except Exception as exc:  # noqa: BLE001 - malformed packet, log & skip
                report.rejected += 1
                report.rejection_reasons.append(f"malformed packet: {exc}")

        rows = aggregator.finalize_flows()
        df = pd.DataFrame(rows)
        logger.info(
            "load_pcap(%s): accepted=%d rejected=%d flows=%d",
            path, report.accepted, report.rejected, len(rows),
        )
        return df, report

    # ------------------------------------------------------------------
    # Live capture (real-time)
    # ------------------------------------------------------------------
    def stream_live(self, interface: str, flow_timeout: float = 5.0) -> Iterator[dict]:
        """
        Generator that yields one feature-dict per completed flow, sniffed
        live from `interface`. Requires scapy + appropriate OS packet
        capture permissions (root / CAP_NET_RAW on Linux).
        """
        try:
            from scapy.all import sniff  # noqa: F401
        except ImportError as exc:
            raise IngestionError(
                "scapy is required for live capture. Install with "
                "'pip install scapy'."
            ) from exc

        from .feature_extractor import PacketFlowAggregator
        aggregator = PacketFlowAggregator(flow_timeout=flow_timeout)

        def _on_packet(pkt):
            try:
                aggregator.add_packet(pkt)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Dropped malformed live packet: %s", exc)
            for flow in aggregator.pop_expired_flows():
                yield flow  # NOTE: illustrative; see scripts/run_live.py for
                            # the production loop that drains this via a queue

        from scapy.all import sniff
        logger.info("Starting live capture on interface=%s", interface)
        sniff(iface=interface, prn=_on_packet, store=False)
