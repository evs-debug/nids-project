import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data import make_preprocessor


class RuleBaseline:
    """Hand-written threshold rules in the style of a signature IDS. Never sees labels."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        pred = np.zeros(len(X), dtype=int)
        # later assignments win, so the rarer and more specific classes go last
        pred[self._probe(X)] = 2
        pred[self._dos(X)] = 1
        pred[self._r2l(X)] = 3
        pred[self._u2r(X)] = 4
        return pred

    @staticmethod
    def _dos(X):
        syn_flood = (X["serror_rate"] > 0.8) & (X["count"] > 20)
        reject_flood = (X["rerror_rate"] > 0.8) & (X["count"] > 100)
        icmp_flood = (X["protocol_type"] == "icmp") & (X["count"] > 20)
        oversized_http = (X["service"] == "http") & (X["src_bytes"] > 50000)
        malformed = (X["wrong_fragment"] > 0) | (X["land"] == 1)
        return (syn_flood | reject_flood | icmp_flood | oversized_http | malformed).to_numpy()

    @staticmethod
    def _probe(X):
        icmp_sweep = (X["protocol_type"] == "icmp") & (X["srv_diff_host_rate"] > 0.5)
        port_scan = (X["dst_host_diff_srv_rate"] > 0.6) & (X["dst_host_rerror_rate"] > 0.2)
        many_services = X["diff_srv_rate"] > 0.5
        return (icmp_sweep | port_scan | many_services).to_numpy()

    @staticmethod
    def _r2l(X):
        failed_logins = X["num_failed_logins"] > 0
        guest_ftp = (X["is_guest_login"] == 1) & X["service"].isin(["ftp", "ftp_data"])
        return (failed_logins | guest_ftp).to_numpy()

    @staticmethod
    def _u2r(X):
        return ((X["root_shell"] == 1) | (X["su_attempted"] > 0) | (X["num_shells"] > 0)).to_numpy()


def build_logreg():
    return make_pipeline(make_preprocessor(), LogisticRegression(max_iter=2000))
