"""Console-script adapters."""

from .errors import MSSTError


def inference() -> None:
    """Run folder inference using command-line arguments."""

    from .inference import proc_folder

    try:
        proc_folder()
    except (MSSTError, FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


def train() -> None:
    """Run model training using command-line arguments."""

    from .training import train_model

    try:
        train_model()
    except (MSSTError, FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


def valid() -> None:
    """Run the historical validation CLI through the packaged runtime."""

    from ._validation import check_validation

    try:
        check_validation(None)
    except (MSSTError, FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
