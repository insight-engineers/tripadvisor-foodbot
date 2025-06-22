import time
from typing import Dict

import numba as nb
import numpy as np
import pandas as pd
from scipy.stats import rankdata


@nb.njit
def compute_concordance(a, b, weights, sum_weights, q_arr, p_arr):
    c = 0.0
    for j in range(a.shape[0]):
        diff = b[j] - a[j]
        if diff <= q_arr[j]:
            c_j = 1.0
        elif diff > p_arr[j]:
            c_j = 0.0
        else:
            c_j = (p_arr[j] - diff) / (p_arr[j] - q_arr[j])
        c += weights[j] * c_j
    return c / sum_weights


@nb.njit
def compute_discordance(a, b, p_arr, v_arr, out):
    for j in range(a.shape[0]):
        diff = b[j] - a[j]
        if diff <= p_arr[j]:
            out[j] = 0.0
        elif diff > v_arr[j]:
            out[j] = 1.0
        else:
            out[j] = (diff - p_arr[j]) / (v_arr[j] - p_arr[j])
    return out


@nb.njit
def compute_credibility(c, d, weights, sum_weights):
    cr = c
    for j in range(d.shape[0]):
        if d[j] > c:
            cr *= (1.0 - d[j]) / (1.0 - c)
    return cr if cr > 0 else 0.0


@nb.njit
def build_outranking(A, weights, sum_weights, q_arr, p_arr, v_arr):
    n, m = A.shape
    outr = np.zeros((n, n), dtype=np.float64)
    d_vec = np.empty(m, dtype=np.float64)
    for i in range(n):
        a = A[i]
        for j in range(n):
            if i == j:
                continue
            b = A[j]
            c = compute_concordance(a, b, weights, sum_weights, q_arr, p_arr)
            d = compute_discordance(a, b, p_arr, v_arr, d_vec)
            outr[i, j] = compute_credibility(c, d, weights, sum_weights)
    return outr


def build_electre_iii(
    dataframe: pd.DataFrame,
    user_preferences: Dict[str, float],
    thresholds: Dict[str, Dict[str, float]],
):
    """
    Build the Electre III ranking dataframe.
    Example params:

    user_preferences = {
        'food_score': 0.1,
        'ambience_score': 0.15,
        'price_score': 0.03,
        'service_score': 0.25,
        'location_score': 0.05,
        'distance_score': 0.02,
        'query_matching_score': 0.4
    }

    thresholds = {
        'food_score': {'q': 0.05, 'p': 0.10, 'v': 0.20},
        'ambience_score': {'q': 0.05, 'p': 0.10, 'v': 0.20},
        'price_score': {'q': 0.05, 'p': 0.10, 'v': 0.20},
        'service_score': {'q': 0.05, 'p': 0.10, 'v': 0.20},
        'location_score': {'q': 0.05, 'p': 0.10, 'v': 0.20},
        'distance_score': {'q': 0.05, 'p': 0.10, 'v': 0.20},
        'query_matching_score': {'q': 0.05, 'p': 0.10, 'v': 0.20}
    }
    """
    try:
        dataframe = dataframe.copy()
        score_columns = list(user_preferences.keys())
        weights_arr = np.array([user_preferences[c] for c in score_columns], dtype=np.float64)
        sum_w = weights_arr.sum()
        q_arr = np.array([thresholds[c]["q"] for c in score_columns], dtype=np.float64)
        p_arr = np.array([thresholds[c]["p"] for c in score_columns], dtype=np.float64)
        v_arr = np.array([thresholds[c]["v"] for c in score_columns], dtype=np.float64)
        A = dataframe[score_columns].to_numpy(dtype=np.float64)
        outranking = build_outranking(A, weights_arr, sum_w, q_arr, p_arr, v_arr)
        net_cred = outranking.sum(axis=1) - outranking.sum(axis=0)
        dataframe["electre_score"] = net_cred
        dataframe["electre_rank"] = rankdata(-net_cred, method="min")
        return dataframe
    except Exception as e:
        print(f"Error in build_electre_iii: {e}")
        return pd.DataFrame()
    finally:
        time.sleep(1.5)
