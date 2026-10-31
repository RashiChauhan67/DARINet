import numpy as np
import pytest

from darinet.model import build_darinet


@pytest.mark.parametrize(
    ("num_classes", "expected"),
    [(5, 648_390), (3, 648_132), (4, 648_261)],  # Cassava, Rice, Maize: totals from model.summary()
)
def test_parameter_count_matches_original_notebooks(num_classes, expected):
    assert build_darinet(num_classes=num_classes).count_params() == expected


def test_forward_pass_gives_a_probability_distribution():
    model = build_darinet(input_shape=(32, 32, 3), num_classes=5)
    out = model.predict(np.random.rand(3, 32, 32, 3).astype("float32"), verbose=0)
    assert out.shape == (3, 5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)
