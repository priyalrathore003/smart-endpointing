"""The committed raw responses must rebuild exactly the dataset the manifest describes."""

import json
from pathlib import Path

import pytest

from endpointing.data.build import BuildConfig, build

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifests" / "synth_v1.json"


@pytest.mark.skipif(not MANIFEST.exists() or not any(RAW.glob("*.json")), reason="synth_v1 not present")
def test_synth_v1_rebuilds_to_the_manifest_hash(tmp_path):
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rebuilt = build(RAW, tmp_path, "synth_v1", BuildConfig(**expected["build_config"]))
    assert rebuilt["data_hash"] == expected["data_hash"]
    assert rebuilt["counts"] == expected["counts"]
    assert rebuilt["raw"]["batches"] == 480
