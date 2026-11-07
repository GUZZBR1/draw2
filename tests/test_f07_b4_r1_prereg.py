import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = REPO_ROOT / "docs/F07_B4_R1_PRE_REGISTRO_2026-11-07.json"
PROTOCOL_PATH = REPO_ROOT / "docs/F07_B4_R1_PRE_REGISTRO_2026-11-07.md"


def test_r1_uses_new_seed_and_output_namespace_without_authorizing_execution():
    registration = json.loads(REGISTRATION_PATH.read_text())
    auth = registration["authorization"]

    assert registration["protocol_id"] == "F-07-B4-R1-instrument-handoff"
    assert registration["version"] == 5
    assert registration["sampling"]["block_seed"].startswith(
        "SeedSequence([20261107, 5, profile_id, phase_id, block_id])"
    )
    assert registration["runner"]["absent_outputs_required_before_start"] == [
        "instrument-calibration-v5.json",
        "frozen-instrument-threshold-v5.json",
        "instrument-validation-v5.json",
        "raster-diagnostic-v5.json",
        "preflight-v5.json",
    ]
    assert re.fullmatch(r"[0-9a-f]{64}", registration["runner"]["sha256"])
    assert registration["sampling"]["blocks"]["calibration_per_profile"] == 581
    assert registration["sampling"]["blocks"]["validation_per_profile"] == 581
    assert all(
        auth[key] is False
        for key in (
            "instrument_calibration_authorized",
            "raster_diagnostic_authorized",
            "preflight_authorized",
            "execution_authorized",
            "confirmation_authorized",
            "mirror_approved",
        )
    )
    assert registration["raster_decomposition"]["authorized"] is False
    assert registration["future_H1_preflight"]["authorized"] is False
    assert registration["grid_and_scope"]["grid_authorized"] is False
    assert registration["grid_and_scope"]["confirmation_authorized"] is False


def test_r1_records_v4_handoff_failure_and_freezes_profile_initialization():
    registration = json.loads(REGISTRATION_PATH.read_text())
    protocol = PROTOCOL_PATH.read_text()
    prior = registration["prior_v4_instrument_attempt"]
    state_contract = registration["sampling"]["state_per_geometry_block"]

    assert prior["artifact_name"] == "instrument-calibration-v4.json"
    assert prior["status"] == "failed_or_incomplete"
    assert prior["excluded_from_calibration"] is True
    assert "app.hand_state=None" in prior["interpretation"]
    assert "Activate the selected profile's hand state" in state_contract[1]
    assert "before its first frame" in state_contract[1]
    assert "never mutate the shared base" in state_contract[1]
    assert registration["runner"]["sha256"] in protocol
    assert "instrument-calibration-v4.json" in protocol
    assert "não evidência" in protocol
    assert "não contém um registro N0 separado" in protocol
