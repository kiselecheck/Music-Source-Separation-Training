"""Compatibility wrapper around the public metadata API."""

from msst import build_metadata
from msst.utils.settings import parse_args_train


args = parse_args_train(None)
result = build_metadata(
    config_path=args.config_path,
    data_path=args.data_path,
    results_path=args.results_path,
    model_type=args.model_type,
    dataset_type=args.dataset_type,
    num_workers=args.num_workers or None,
)
print(f"Metadata: {result.path}")
print(f"Track entries: {result.track_count}")
