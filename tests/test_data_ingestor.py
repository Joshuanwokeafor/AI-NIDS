"""tests/test_data_ingestor.py — unit tests for Stage 1 (DataIngestor)."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestor import DataIngestor, IngestionError  # noqa: E402


@pytest.fixture
def valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    path.write_text(
        "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,"
        "wrong_fragment,urgent,hot,num_failed_logins,logged_in,"
        "num_compromised,root_shell,su_attempted,num_root,"
        "num_file_creations,num_shells,num_access_files,num_outbound_cmds,"
        "is_host_login,is_guest_login,count,srv_count,serror_rate,"
        "srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,"
        "diff_srv_rate,srv_diff_host_rate,dst_host_count,"
        "dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,"
        "dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,"
        "dst_host_serror_rate,dst_host_srv_serror_rate,"
        "dst_host_rerror_rate,dst_host_srv_rerror_rate,class,difficulty\n"
        "0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,"
        "0,0,0,0,1,0,0,9,9,1,0,0.11,0,0,0,0,0,normal,20\n"
        "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,10,"
        "1,1,0,0,0.04,0.06,0,255,10,0.04,0.06,0,0,1,1,0,0,neptune,21\n"
    )
    return str(path)


@pytest.fixture
def malformed_csv(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text(
        "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,"
        "wrong_fragment,urgent,hot,num_failed_logins,logged_in,"
        "num_compromised,root_shell,su_attempted,num_root,"
        "num_file_creations,num_shells,num_access_files,num_outbound_cmds,"
        "is_host_login,is_guest_login,count,srv_count,serror_rate,"
        "srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,"
        "diff_srv_rate,srv_diff_host_rate,dst_host_count,"
        "dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,"
        "dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,"
        "dst_host_serror_rate,dst_host_srv_serror_rate,"
        "dst_host_rerror_rate,dst_host_srv_rerror_rate,class,difficulty\n"
        "NOT_A_NUMBER,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,"
        "0,0,0,0,1,0,0,9,9,1,0,0.11,0,0,0,0,0,normal,20\n"
        "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,10,"
        "1,1,0,0,0.04,0.06,0,255,10,0.04,0.06,0,0,1,1,0,0,neptune,21\n"
        "1,2,3\n"  # wrong column count
    )
    return str(path)


class TestDataIngestor:
    def test_load_valid_csv_accepts_all_rows(self, valid_csv):
        ingestor = DataIngestor()
        df, report = ingestor.load_csv(valid_csv)
        assert report.accepted == 2
        assert report.rejected == 0
        assert len(df) == 2
        assert "class" in df.columns
        assert set(df["class"]) == {"normal", "neptune"}

    def test_load_csv_missing_file_raises(self):
        ingestor = DataIngestor()
        with pytest.raises(IngestionError):
            ingestor.load_csv("/nonexistent/path/does_not_exist.csv")

    def test_load_csv_rejects_malformed_rows(self, malformed_csv):
        ingestor = DataIngestor()
        df, report = ingestor.load_csv(malformed_csv)
        # 1 row has non-numeric duration, 1 row has wrong column count -> 2 rejected
        assert report.rejected == 2
        assert report.accepted == 1
        assert len(report.rejection_reasons) == 2

    def test_load_csv_empty_file_raises(self, tmp_path):
        empty_path = tmp_path / "empty.csv"
        empty_path.write_text("")
        ingestor = DataIngestor()
        with pytest.raises(IngestionError):
            ingestor.load_csv(str(empty_path))

    def test_coerced_numeric_dtypes(self, valid_csv):
        ingestor = DataIngestor()
        df, _ = ingestor.load_csv(valid_csv)
        assert pd.api.types.is_numeric_dtype(df["duration"])
        assert pd.api.types.is_numeric_dtype(df["src_bytes"])
        assert not pd.api.types.is_numeric_dtype(df["protocol_type"])

    def test_load_pcap_missing_file_raises(self):
        ingestor = DataIngestor()
        with pytest.raises(IngestionError):
            ingestor.load_pcap("/nonexistent/file.pcap")

    def test_ingestion_report_as_dict(self, valid_csv):
        ingestor = DataIngestor()
        _, report = ingestor.load_csv(valid_csv)
        d = report.as_dict()
        assert d["accepted"] == 2
        assert d["rejected"] == 0
        assert isinstance(d["rejection_reasons"], list)
