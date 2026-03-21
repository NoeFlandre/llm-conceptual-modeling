import pandas as pd


def assert_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str] | set[str],
    *,
    label: str = "columns",
) -> None:
    missing_columns = sorted(set(required_columns).difference(dataframe.columns))
    if missing_columns:
        msg = f"Missing required {label}: {missing_columns}"
        raise ValueError(msg)


def assert_output_columns(
    dataframe: pd.DataFrame,
    expected_columns: list[str],
) -> None:
    actual_columns = list(dataframe.columns)
    if actual_columns != expected_columns:
        msg = f"Unexpected output columns: expected {expected_columns}, got {actual_columns}"
        raise ValueError(msg)
