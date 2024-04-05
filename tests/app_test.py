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
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def mock_load_model(*args, **kwargs):
    pass


with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config", overrides=["cache=local", "db=test"])

db_url = f"postgresql://{cfg.db.user}:{cfg.db.password}@{cfg.db.host}:{cfg.db.port}/{cfg.db.name}"
engine = create_engine(
    db_url,
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = create_app(cfg)
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestDependencies:

    model = Model(cfg.app.model_name)
    df = pd.read_csv(f"{Path(__file__).parent}/test_texts.csv").squeeze("columns")

    def test_load_model(self):
        assert isinstance(self.model.model, Pipeline)

    @pytest.mark.parametrize("threshold", [False, 0.5])
    @pytest.mark.parametrize(
        "return_labels, return_names, expected",
        [(True, True, (10, 2)), (True, False, (10,)), (False, True, (10,))],
    )
    @pytest.mark.dependency(depends=["test_load_model"])
    def test_predict(self, threshold, return_labels, return_names, expected):
        prediction = self.model.predict(
            self.df,
            threshold=threshold,
            return_labels=return_labels,
            return_names=return_names,
        )
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == expected

    def test_predict_labels(self):
        prediction = self.model.predict(self.df, return_labels=True, return_names=False)
        assert prediction.dtype == "int64"
        assert np.all((prediction >= -1) & (prediction <= 9))

    def test_predict_names(self):
        prediction = self.model.predict(self.df, return_labels=False, return_names=True)
        assert np.all([name in label2name.values() for name in prediction])


@patch("app.dependencies.load_model", mock_load_model)
class TestUsers:

    @pytest.fixture(autouse=True)
    def test_db(self):
        Base.metadata.create_all(bind=engine)
        yield
        Base.metadata.drop_all(bind=engine)

    @pytest.fixture()
    def add_faux_users(self, request):
        for uid in [1, 2]:
            client.post("/users/", json={"id": uid})
        if request.param == "active":
            for i, requests, rating in zip([1, 2], [3, 5], [5, 4], strict=True):
                for _ in range(requests):
                    client.patch(f"/users/{i}/requests")
                client.patch(
                    f"/users/{i}/{rating}",
                )

        return

    def test_root(self):
        response = client.get("/users/")

        assert response.status_code == 200
        assert response.json() == {
            "message": "Welcome to the Authorship Identification Service"
        }

    def test_create_user(self):
        response = client.post("/users/", json={"id": 3})
        data = response.json()

        assert response.status_code == 200
        assert data["id"] == 3
        assert data["requests"] == 0
        assert data["rating"] is None

    @pytest.mark.parametrize("add_faux_users", ["active"], indirect=["add_faux_users"])
    def test_create_registered_user(self, add_faux_users):
        response = client.post("/users/", json={"id": 1})
        assert response.status_code == 400
        assert response.json() == {"detail": "User already registered"}

    @pytest.mark.parametrize("add_faux_users", ["passive"], indirect=["add_faux_users"])
    def test_update_user_requests(self, add_faux_users):
        response = client.patch(
            "/users/2/requests",
        )
        data = response.json()

        assert response.status_code == 200
        assert data["id"] == 2
        assert data["requests"] == 1

    def test_update_unregistered_user_requests(self):
        response = client.patch(
            "/users/1/requests",
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "User not found"}

    @pytest.mark.parametrize("add_faux_users", ["active"], indirect=["add_faux_users"])
    def test_update_user_rating(self, add_faux_users):
        response = client.patch(
            "/users/2/5",
        )
        data = response.json()

        assert response.status_code == 200
        assert data["id"] == 2
        assert data["rating"] == 5

    def test_update_unregistered_user_rating(self):
        response = client.patch(
            "/users/1/5",
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
            response = client.get("/users/stats")
            data = response.json()

            assert response.status_code == 200
            assert data["total_users"] == 2
            assert data["total_requests"] == expected_requests
            assert data["avg_rating"] == expected_rating


@patch("app.dependencies.load_model", mock_load_model)
class TestItems:

    @patch.object(
        Model, "predict", return_value=np.array([[0, "А. Пушкин"]], dtype=object)
    )
    def test_predict_text(self, mock_predict):
        with TestClient(app) as client:
            response = client.post("/items/predict_text", json={"text": "some text"})
            data = response.json()

            assert response.status_code == 200
            assert data["label"] == 0
            assert data["name"] == "А. Пушкин"

    @patch.object(
        Model, "predict", return_value=np.array(["А. Пушкин"] * 10, dtype=object)
    )
    def test_predict_texts(self, mock_predict):
        with TestClient(app) as client:
            with open(f"{Path(__file__).parent}/test_texts.csv", "rb") as f:
                response = client.post(
                    "/items/predict_texts",
                    files={"file": ("test_texts.csv", f, "text/csv")},
                )

                assert response.status_code == 200
                assert (
                    "filename=test_texts.csv" in response.headers["content-disposition"]
                )
                assert "text/csv" in response.headers["content-type"]
