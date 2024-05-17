# File: tests/test_database.py

from unittest.mock import MagicMock, patch

import pytest

from HappyChoicesAI.ai_state import (
    Database,
    HistoricalExample,
)  # Adjust the import as needed


@pytest.fixture
def db():
    return Database(
        host="localhost", database="happychoices", user="user", password="password"
    )


def test_get_all_historical_examples(db):
    mock_records = [
        (1, "Situation 1", "Action 1", "Reasoning 1"),
        (2, "Situation 2", "Action 2", "Reasoning 2"),
    ]
    expected_examples = [
        HistoricalExample(
            situation="Situation 1", action_taken="Action 1", reasoning="Reasoning 1"
        ),
        HistoricalExample(
            situation="Situation 2", action_taken="Action 2", reasoning="Reasoning 2"
        ),
    ]

    with patch("mysql.connector.connect") as mock_connect:
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_records
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        examples = db.get_all_historical_examples()

        assert examples == expected_examples


def test_get_connection(db):
    with patch("mysql.connector.connect") as mock_connect:
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_connection.is_connected.return_value = True

        connection = db.get_connection()

        assert connection == mock_connection
        mock_connect.assert_called_once_with(
            host=db.host, database=db.database, user=db.user, password=db.password
        )


if __name__ == "__main__":
    pytest.main()
