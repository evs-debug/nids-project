from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL + ["label", "difficulty"]]

CLASSES = ["normal", "DoS", "Probe", "R2L", "U2R"]

_FAMILIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
            "udpstorm", "processtable", "worm", "mailbomb"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
            "snmpguess", "xlock", "xsnoop", "httptunnel"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps",
            "sqlattack", "xterm"],
}
ATTACK_TO_CLASS = {"normal": "normal"}
ATTACK_TO_CLASS.update(
    {attack: family for family, attacks in _FAMILIES.items() for attack in attacks}
)


def load_split(name):
    df = pd.read_csv(RAW_DIR / f"KDD{name}+.txt", header=None, names=COLUMNS)
    family = df["label"].map(ATTACK_TO_CLASS)
    unmapped = df.loc[family.isna(), "label"].unique()
    if len(unmapped):
        raise ValueError(f"attack labels with no class: {list(unmapped)}")
    y = family.map(CLASSES.index)
    X = df.drop(columns=["label", "difficulty"])
    return X, y


def make_preprocessor():
    # the test split has services that never appear in train
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
