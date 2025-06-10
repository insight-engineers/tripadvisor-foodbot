import pandas as pd

from src.helper.utils import haversine_distance


def compute_criterion_score(positive: pd.Series, negative: pd.Series) -> pd.Series:
    """
    Compute the criterion score based on the given positive and negative values.
        - Case 1: If there are more positive than negative values, the score is computed as (1 - negative / positive).
        - Case 2: If there are more negative than positive values, the score is computed as (positive / negative - 1).
        - If neither case applies, the score is set to 0.
    """
    more_positive = (1 - negative / positive).where(positive > negative)
    more_negative = (positive / negative - 1).where(positive < negative)
    criterion_score = more_positive.combine_first(more_negative).fillna(0)

    return criterion_score


def compute_normalized_criterion_score(df: pd.DataFrame, criteria: list[str]) -> pd.DataFrame:
    """
    Compute the raw and normalized scores for each criterion in the given criteria list
    For query matching score, we multiply by 100 because cosine similarity is in range [0, 1]
    """
    df_normalized_scores = df.copy()

    for criterion in criteria:
        pos_col = f"{criterion}_positive"
        neg_col = f"{criterion}_negative"
        raw_score_col = f"{criterion}_score_raw"
        df_normalized_scores[raw_score_col] = compute_criterion_score(df[pos_col], df[neg_col])
        norm_score_col = f"{criterion}_score_normalized"
        df_normalized_scores[norm_score_col] = (df_normalized_scores[raw_score_col] + 1) * 50

    relevant_cols = [
        "location_id",
        "location_name",
        "location_overall_rate",
        "review_count",
        "latitude",
        "longitude",
        "query_matching_score",
    ] + [f"{c}_score_normalized" for c in criteria]

    df_normalized_scores = df_normalized_scores[relevant_cols]
    df_normalized_scores = df_normalized_scores.rename(
        columns={col: col.replace("_normalized", "") for col in df_normalized_scores.columns}
    )
    df_normalized_scores["query_matching_score"] = df_normalized_scores["query_matching_score"] * 100

    return df_normalized_scores


def compute_distance_score(df: pd.DataFrame, max_distance: float, user_lat: float, user_long: float) -> pd.DataFrame:
    """
    Compute the distance score based on the given dataframe and max_distance.
    """
    df["distance"] = df.apply(
        lambda row: haversine_distance(row["latitude"], row["longitude"], user_lat, user_long),
        axis=1,
    )
    df["distance_score"] = 100 * (1 - df["distance"].clip(upper=max_distance) / max_distance)
    return df
