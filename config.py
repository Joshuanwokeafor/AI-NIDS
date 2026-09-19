"""
config.py
---------
Central configuration for the AI-NIDS system: dataset schema, feature
lists, model hyperparameters, and runtime thresholds. Keeping these in
one place satisfies the maintainability (NF4) requirement and lets the
report cite exact hyperparameters used for the empirical results.
"""

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
# Overridable so tests can isolate their database and hosted deployments
# can point at a writable location (e.g. /tmp on serverless platforms).
DB_PATH = os.environ.get("NIDS_DB_PATH", os.path.join(BASE_DIR, "db", "nids.sqlite3"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# NSL-KDD schema (41 base features + 'class' + 'difficulty')
# --------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "class", "difficulty",
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [
    c for c in NSL_KDD_COLUMNS
    if c not in CATEGORICAL_FEATURES + ["class", "difficulty"]
]

# Mapping from the ~39 raw NSL-KDD attack labels to the 5 coarse classes
# required by FR3. Anything not listed defaults to "Normal" only if the
# literal label is "normal"; unseen attack strings fall back to "Probe"
# is WRONG - unseen labels are logged and mapped to "Unknown" instead so
# they do not silently corrupt training statistics.
ATTACK_CATEGORY_MAP = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L (Remote to Local)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L",
    "sendmail": "R2L", "named": "R2L",
    # U2R (User to Root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}

CLASS_LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_LABELS)}

# --------------------------------------------------------------------------
# Preprocessing (Stage 2)
# --------------------------------------------------------------------------
ONE_HOT_EXPANDED_FEATURE_COUNT = 122  # 41 base -> 122 after one-hot (protocol=3, service=70, flag=11 typical)
SMOTE_TARGET_RATIO = 1.0             # 1:1 minority:majority after SMOTE
SMOTE_K_NEIGHBORS = 5

# --------------------------------------------------------------------------
# Feature selection (Stage 3)
# --------------------------------------------------------------------------
TOP_K_FEATURES = 40

# --------------------------------------------------------------------------
# Stage 4: Classifier hyperparameters
# --------------------------------------------------------------------------
SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   random_state=42)

RF_PARAMS = dict(n_estimators=100, max_depth=None, n_jobs=-1,
                  random_state=42, class_weight="balanced")

LSTM_PARAMS = dict(
    units=64,
    dropout=0.3,
    recurrent_dropout=0.2,
    dense_units=32,
    learning_rate=1e-3,
    batch_size=256,
    epochs=15,
    timesteps=1,   # each NSL-KDD record is treated as a single timestep
                   # sequence of length `sequence_window` (see LSTMClassifier)
    sequence_window=10,  # number of consecutive records grouped as one
                          # temporal sequence, addressing the "temporal
                          # awareness" literature gap
)

# --------------------------------------------------------------------------
# Stage 5: Fuzzy aggregation
# --------------------------------------------------------------------------
FUZZY_N_CLUSTERS = 5           # matches the 5 output classes
FUZZY_M = 2.0                  # fuzziness exponent
FUZZY_ERROR = 1e-5
FUZZY_MAX_ITER = 1000
AMBIGUITY_THRESHOLD_DELTA = 0.6  # below this max-membership -> analyst queue

# --------------------------------------------------------------------------
# Web / RBAC
# --------------------------------------------------------------------------
SECRET_KEY = os.environ.get("NIDS_SECRET_KEY", "dev-secret-change-me")
ROLES = ("admin", "analyst")
