import csv

from msst._validation import write_validation_track_metrics_csv


def test_validation_track_csv_appends_track_rows(tmp_path):
    per_track_metrics = {
        "/valid/hard_positive_valid/track_a/mixture.wav": {
            "synth_bass": {"sdr": 3.5, "bleedless": 92.0},
        },
        "/valid/hard_negative_valid/track_b/mixture.wav": {
            "synth_bass": {"sdr": -1.25, "bleedless": 88.0},
        },
    }

    output_path = write_validation_track_metrics_csv(str(tmp_path), 7, per_track_metrics)
    with open(output_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["epoch"] == "7"
    assert rows[0]["validation_group"] == ""
    assert rows[0]["track_id"] == "track_a"
    assert rows[0]["instrument"] == "synth_bass"
    assert rows[0]["sdr"] == "3.5"
    assert rows[1]["validation_group"] == ""
    assert rows[1]["sdr"] == "-1.25"

    write_validation_track_metrics_csv(str(tmp_path), 8, per_track_metrics)
    with open(output_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["epoch"] for row in rows] == ["7", "7", "8", "8"]


def test_validation_track_csv_replaces_resumed_epoch_and_preserves_metric_columns(tmp_path):
    mixture_path = "/valid/collection/track_a/mixture.wav"
    output_path = write_validation_track_metrics_csv(
        str(tmp_path), 3, {mixture_path: {"vocals": {"sdr": 1.0}}},
        valid_paths=["/valid/collection"],
    )
    write_validation_track_metrics_csv(
        str(tmp_path), 3, {mixture_path: {"vocals": {"sdr": 2.0}}},
        valid_paths=["/valid/collection"],
    )
    write_validation_track_metrics_csv(
        str(tmp_path), 4, {mixture_path: {"vocals": {"bleedless": 90.0}}},
        valid_paths=["/valid/collection"],
    )

    with open(output_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert rows[0]["sdr"] == "2.0"
    assert rows[0]["track_key"] == "collection/track_a"
    assert rows[1]["sdr"] == ""
    assert rows[1]["bleedless"] == "90.0"


def test_validation_track_csv_uses_explicit_key_and_checkpoint(tmp_path):
    from msst._validation_manifest import ValidationTrack

    mixture_path = str(tmp_path / "valid" / "track_a" / "mixture.wav")
    output_path = str(tmp_path / "scores.csv")
    info = {mixture_path: ValidationTrack("dataset/track_a", mixture_path, "holdout")}
    for checkpoint, score in (("a.ckpt", 1.0), ("b.ckpt", 2.0)):
        write_validation_track_metrics_csv(
            "", None, {mixture_path: {"vocals": {"sdr": score}}},
            output_path=output_path, track_info=info, checkpoint_path=checkpoint,
        )
    with open(output_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["track_key"] for row in rows] == ["dataset/track_a"] * 2
    assert [row["validation_group"] for row in rows] == ["holdout"] * 2
    assert [row["checkpoint_path"] for row in rows] == ["a.ckpt", "b.ckpt"]


def test_validation_track_csv_keeps_rows_from_older_header(tmp_path):
    mixture_path = str(tmp_path / "valid" / "track_a" / "mixture.wav")
    report = tmp_path / "validation_metrics_by_track.csv"
    with report.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "validation_group", "track_id", "mixture_path", "instrument", "sdr"],
        )
        writer.writeheader()
        writer.writerow({
            "epoch": "1", "validation_group": "regular_valid", "track_id": "track_a",
            "mixture_path": mixture_path, "instrument": "vocals", "sdr": "1.5",
        })

    write_validation_track_metrics_csv(
        str(tmp_path), 2, {mixture_path: {"vocals": {"bleedless": 90.0}}},
        valid_paths=[str(tmp_path / "valid")],
    )
    with report.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[0]["track_key"] == "valid/track_a"
    assert rows[0]["sdr"] == "1.5"
    assert rows[1]["bleedless"] == "90.0"
