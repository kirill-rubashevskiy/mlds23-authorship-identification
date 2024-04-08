from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from hydra import compose, initialize
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.dependencies import Model, get_db
from app.main import create_app
from mlds23_authorship_identification.utils import label2name


def override_get_db():
    """Overrides database session with the test database session."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def mock_load_model(*args, **kwargs):
    """Overrides load model from S3 for the tests where model is not used."""
    pass


# initialize Hydra and compose a config
with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config", overrides=["cache=local", "db=test"])


# create test database session
db_url = f"postgresql://{cfg.db.user}:{cfg.db.password}@{cfg.db.host}:{cfg.db.port}/{cfg.db.name}"
engine = create_engine(
    db_url,
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


app = create_app(cfg)
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestModel:
    """
    Tests for app.dependencies.Model class.

    Attributes:
        model: an app.dependencies.Model instance.
        df: a pandas.Series with texts to test the model against.
    """

    model = Model(cfg.app.model_name)
    df = pd.read_csv(f"{Path(__file__).parent}/test_texts.csv").squeeze("columns")

    def test_load_model(self):
        assert isinstance(self.model.model, Pipeline)

    @pytest.mark.parametrize("threshold", [False, 0.5])
    @pytest.mark.parametrize(
        "return_labels, return_names, expected",
        [(True, True, (10, 2)), (True, False, (10,)), (False, True, (10,))],
    )
    def test_predict(self, threshold, return_labels, return_names, expected):
        """Tests model prediction type and shape."""

        prediction = self.model.predict(
            self.df,
            threshold=threshold,
            return_labels=return_labels,
            return_names=return_names,
        )
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == expected

    def test_predict_labels(self):
        """Tests model prediction with labels."""

        prediction = self.model.predict(self.df, return_labels=True, return_names=False)
        assert prediction.dtype == "int64"
        assert np.all((prediction >= -1) & (prediction <= 9))

    def test_predict_names(self):
        """Tests model prediction with names."""

        prediction = self.model.predict(self.df, return_labels=False, return_names=True)
        assert np.all([name in label2name.values() for name in prediction])


@patch("app.dependencies.load_model", mock_load_model)
class TestUsers:
    """Tests for users-related endpoints.

    Attributes:
        prefix: prefix of all users-related endpoints.
    """

    prefix = "/users"

    @pytest.fixture(autouse=True)
    def reset_db(self):
        """Resets the test database between tests."""

        Base.metadata.create_all(bind=engine)
        yield
        Base.metadata.drop_all(bind=engine)

    @pytest.fixture()
    def add_faux_users(self, request) -> None:
        """
        Adds faux users to the test database.

        Args:
            request: if request.param == "active", will generate active faux users (making requests, rating the
            service).
        """

        for id in [1, 2]:
            client.post(f"{self.prefix}/", json={"id": id})
        if request.param == "active":
            for id, requests, rating in zip([1, 2], [3, 5], [5, 4], strict=True):
                for _ in range(requests):
                    client.patch(f"{self.prefix}/{id}/requests")
                client.patch(
                    f"{self.prefix}/{id}/{rating}",
                )

        return

    def test_root(self):
        response = client.get(f"{self.prefix}/")
        assert response.status_code == 200
        assert response.json() == {
            "message": "Welcome to the Authorship Identification Service"
        }

    @pytest.mark.parametrize("uid", [3])
    def test_create_user(self, uid):
        response = client.post(f"{self.prefix}/", json={"id": uid})
        data = response.json()
        assert response.status_code == 200
        assert data["id"] == uid
        assert data["requests"] == 0
        assert data["rating"] is None

    @pytest.mark.parametrize("add_faux_users", ["active"], indirect=["add_faux_users"])
    def test_create_registered_user(self, add_faux_users):
        response = client.post(f"{self.prefix}/", json={"id": 1})
        assert response.status_code == 400
        assert response.json() == {"detail": "User already registered"}

    @pytest.mark.parametrize(
        "add_faux_users, uid", [("passive", 2)], indirect=["add_faux_users"]
    )
    def test_update_user_requests(self, add_faux_users, uid):
        response = client.patch(
            f"{self.prefix}/{uid}/requests",
        )
        data = response.json()
        assert response.status_code == 200
        assert data["id"] == uid
        assert data["requests"] == 1

    def test_update_unregistered_user_requests(self):

        response = client.patch(
            f"{self.prefix}/1/requests",
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "User not found"}

    @pytest.mark.parametrize(
        "add_faux_users, uid, rating", [("active", 2, 5)], indirect=["add_faux_users"]
    )
    def test_update_user_rating(self, add_faux_users, uid, rating):
        response = client.patch(
            f"{self.prefix}/{uid}/{rating}",
        )
        data = response.json()
        assert response.status_code == 200
        assert data["id"] == uid
        assert data["rating"] == rating

    def test_update_unregistered_user_rating(self):
        response = client.patch(
            f"{self.prefix}/1/5",
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "User not found"}

    @pytest.mark.parametrize(
        ("add_faux_users", "expected_requests", "expected_rating"),
        [("passive", 0, 0), ("active", 8, 4.5)],
        indirect=["add_faux_users"],
    )
    def test_get_stats(self, add_faux_users, expected_requests, expected_rating):
        with TestClient(app) as client:
            response = client.get(f"{self.prefix}/stats")
            data = response.json()
            assert response.status_code == 200
            assert data["total_users"] == 2
            assert data["total_requests"] == expected_requests
            assert data["avg_rating"] == expected_rating


@patch("app.dependencies.load_model", mock_load_model)
class TestItems:
    """Tests for predictions-related endpoints.

    Attributes:
        prefix: prefix of all predictions-related endpoints.
    """

    prefix = "/items"

    @patch.object(
        Model, "predict", return_value=np.array([[0, "А. Пушкин"]], dtype=object)
    )
    def test_predict_text(self, mock_predict):
        """Tests single text prediction."""

        with TestClient(app) as client:
            response = client.post(
                f"{self.prefix}/predict_text", json={"text": "some text"}
            )
            data = response.json()
            assert response.status_code == 200
            assert data["label"] == 0
            assert data["name"] == "А. Пушкин"

    @patch.object(
        Model, "predict", return_value=np.array(["А. Пушкин"] * 10, dtype=object)
    )
    def test_predict_texts(self, mock_predict):
        """Tests multiple texts predictions."""

        with TestClient(app) as client:
            with open(f"{Path(__file__).parent}/test_texts.csv", "rb") as f:
                response = client.post(
                    f"{self.prefix}/predict_texts",
                    files={"file": ("test_texts.csv", f, "text/csv")},
                )
                assert response.status_code == 200
                assert (
                    "filename=test_texts.csv" in response.headers["content-disposition"]
                )
                assert "text/csv" in response.headers["content-type"]
