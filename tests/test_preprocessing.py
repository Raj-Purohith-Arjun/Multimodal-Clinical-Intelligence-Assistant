import pandas as pd
import pytest

from src.data.preprocessing import validate_manifest


def test_validate_manifest_missing_columns() -> None:
    frame = pd.DataFrame({"patient_id": ["P-1"], "clinical_text": ["abc"]})
    with pytest.raises(ValueError):
        validate_manifest(frame)
