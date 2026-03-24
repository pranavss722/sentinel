"""Tests for the baseline training script."""
import numpy as np


def test_generate_synthetic_data_shape():
    from scripts.train_baseline import generate_synthetic_data
    X, y = generate_synthetic_data()
    assert X.shape == (10_000, 10)
    assert y.shape == (10_000,)


def test_generate_synthetic_data_binary_labels():
    from scripts.train_baseline import generate_synthetic_data
    X, y = generate_synthetic_data()
    assert set(np.unique(y)) == {0, 1}


def test_generate_synthetic_data_deterministic():
    from scripts.train_baseline import generate_synthetic_data
    X1, y1 = generate_synthetic_data()
    X2, y2 = generate_synthetic_data()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_synthetic_data_float_features():
    from scripts.train_baseline import generate_synthetic_data
    X, _ = generate_synthetic_data()
    assert X.dtype == np.float64 or X.dtype == np.float32


def test_train_model_returns_xgboost_classifier():
    from scripts.train_baseline import generate_synthetic_data, train_model
    import xgboost as xgb
    X, y = generate_synthetic_data()
    model = train_model(X, y)
    assert isinstance(model, xgb.XGBClassifier)


def test_train_model_can_predict():
    from scripts.train_baseline import generate_synthetic_data, train_model
    X, y = generate_synthetic_data()
    model = train_model(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert all(p in (0, 1) for p in preds)


def test_train_model_can_predict_proba():
    from scripts.train_baseline import generate_synthetic_data, train_model
    X, y = generate_synthetic_data()
    model = train_model(X, y)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
