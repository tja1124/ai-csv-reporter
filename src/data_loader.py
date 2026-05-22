"""Load and validate CSV data."""

from pathlib import Path

import pandas as pd

from src.config import CSV_ENCODINGS
from src.logger import get_logger

logger = get_logger()

# Friendly messages shown in the terminal for common validation issues.
WARNING_MESSAGES = {
    "duplicate_columns": "Some column names are duplicated. Analysis will continue, but review column names.",
    "missing_headers": "Some columns appear to have missing headers (e.g. 'Unnamed: 0').",
    "all_null_columns": "One or more columns contain only null values and will be skipped in charts.",
}


def _read_csv_with_encoding(path: Path) -> pd.DataFrame:
    """
    Try reading a CSV file with common encodings.

    Raises:
        ValueError: If the file cannot be decoded or parsed.
    """
    last_error: Exception | None = None

    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding)
            logger.info("Loaded CSV with encoding: %s", encoding)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            logger.warning("Failed to read CSV with encoding %s", encoding)
        except pd.errors.EmptyDataError:
            raise ValueError(
                "The CSV file is empty. Please provide a file with a header row and at least one data row."
            ) from last_error
        except pd.errors.ParserError as exc:
            raise ValueError(
                "The CSV file appears to be corrupted or incorrectly formatted. "
                "Check for broken rows, mismatched columns, or invalid separators."
            ) from exc

    raise ValueError(
        "Could not read the CSV file due to an unsupported text encoding. "
        "Try saving the file as UTF-8 and run the report again."
    ) from last_error


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a valid CSV.
    """
    path = Path(file_path)
    logger.info("Loading CSV file: %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}\n"
            "Check the file path and try again."
        )

    if not path.is_file():
        raise ValueError(
            f"The path does not point to a file: {path}\n"
            "Provide a valid CSV file path."
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a .csv file, but received '{path.suffix}'.\n"
            "Convert the file to CSV format or provide a different file."
        )

    try:
        df = _read_csv_with_encoding(path)
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error while loading CSV")
        raise ValueError(
            f"Failed to read CSV file: {path}\n"
            "The file may be corrupted or not a valid CSV."
        ) from exc

    logger.info("CSV loaded successfully with shape %s", df.shape)
    return df


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Validate that a DataFrame is usable for analysis.

    Non-fatal issues are returned as warnings so the pipeline can continue.

    Args:
        df: The DataFrame to validate.

    Returns:
        A list of user-friendly warning messages.

    Raises:
        ValueError: If the DataFrame has fatal validation errors.
    """
    warnings: list[str] = []

    if df is None:
        raise ValueError("No data was loaded from the CSV file.")

    if len(df.columns) == 0:
        raise ValueError(
            "The CSV file has no columns.\n"
            "Make sure the file includes a header row."
        )

    if df.empty:
        raise ValueError(
            "The CSV file has headers but no data rows.\n"
            "Add at least one row of data before generating a report."
        )

    # Missing headers often appear as Unnamed columns when no header row exists.
    unnamed_columns = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_columns:
        warning = WARNING_MESSAGES["missing_headers"]
        warnings.append(warning)
        logger.warning("%s Columns: %s", warning, unnamed_columns)

    # Duplicate names can cause confusing analysis results.
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        warning = WARNING_MESSAGES["duplicate_columns"]
        warnings.append(warning)
        logger.warning("%s Duplicates: %s", warning, duplicated)

    # All-null columns add noise without useful information.
    null_columns = [col for col in df.columns if df[col].isnull().all()]
    if null_columns:
        warning = WARNING_MESSAGES["all_null_columns"]
        warnings.append(warning)
        logger.warning("%s Columns: %s", warning, null_columns)

    if warnings:
        logger.info("Validation completed with %d warning(s)", len(warnings))
    else:
        logger.info("Validation passed with no warnings")

    return warnings
