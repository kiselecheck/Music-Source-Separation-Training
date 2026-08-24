"""Console-script adapters."""

from .errors import MSSTError


def inference() -> None:
    from .inference import proc_folder

    try:
        proc_folder()
    except (MSSTError, FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


def train() -> None:
    from .training import train_model

    try:
        train_model()
    except (MSSTError, FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
