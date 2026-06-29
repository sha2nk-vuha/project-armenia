import pytest
import numpy as np
import cv2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def png_bytes() -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 20:44] = [200, 100, 50]
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@pytest.fixture
def anomaly_map() -> np.ndarray:
    arr = np.zeros((1, 1, 32, 32), dtype=np.float32)
    arr[0, 0, 10:22, 10:22] = 0.9
    return arr
