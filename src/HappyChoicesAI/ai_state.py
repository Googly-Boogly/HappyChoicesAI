from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import mysql.connector
from mysql.connector import Error


@dataclass
class HistoricalExample:
    situation: str
    action_taken: str
    reasoning: str


@dataclass
class EthicistAIState:
    situation: str = ""
    criteria: str = ""
    historical_examples: List[HistoricalExample] = field(default_factory=list)
    thought_experiments: List[Dict[str, str]] = field(default_factory=list)
    best_action: str = ""


class Database:
    def __init__(self, host: str, database: str, user: str, password: str):
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    def get_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            if connection.is_connected():
                return connection
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            return None

    def get_all_historical_examples(self) -> List[HistoricalExample]:
        connection = self.get_connection()
        if not connection:
            return []

        query = "SELECT * FROM historical_examples"
        cursor = connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()

        examples = [
            HistoricalExample(situation=row[1], action_taken=row[2], reasoning=row[3])
            for row in records
        ]

        cursor.close()
        connection.close()
        return examples


# Example usage:
# db = Database(host='mysql', database='happychoices', user='root', password='password')
# historical_examples = db.get_all_historical_examples()
