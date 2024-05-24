from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mysql.connector
from mysql.connector import Error


@dataclass
class HistoricalExample:
    situation: str
    action_taken: str
    reasoning: str


@dataclass
class SummaryAgentAllThoughtExperiments:
    all_thought_experiments: str = ""
    conclusion: str = ""
    insights: str = ""
    historical_examples_summary: str = ""
    themes: str = ""
    chosen_best_action_summary: str = ""
    other_thought_experiments_summary: List[str] = field(default_factory=list)
    introduction: str = ""
    lessons_learned: str = ""
    markdown: str = ""



@dataclass
class EthicistAIState:
    situation: str = ""
    criteria: str = ""
    historical_examples: List[HistoricalExample] = field(default_factory=list)
    thought_experiments: List[Dict[str, str or int]] = field(default_factory=list)
    best_action: Optional[int] = None  # This is the id of the best action

@dataclass
class ModelUsedAndThreadCountState:
    model_used: str = ""
    thread_count: int = 0


class StateManager:
    _instance = None

    @staticmethod
    def get_instance():
        if StateManager._instance is None:
            StateManager()
        return StateManager._instance

    def __init__(self):
        if StateManager._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.state = EthicistAIState()
            StateManager._instance = self


class StateManagerSummary:
    _instance = None

    @staticmethod
    def get_instance():
        if StateManagerSummary._instance is None:
            StateManagerSummary()
        return StateManagerSummary._instance

    def __init__(self):
        if StateManagerSummary._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.state = SummaryAgentAllThoughtExperiments()
            StateManagerSummary._instance = self


class ModelUsedAndThreadCount:
    _instance = None

    @staticmethod
    def get_instance():
        if ModelUsedAndThreadCount._instance is None:
            ModelUsedAndThreadCount()
        return ModelUsedAndThreadCount._instance

    def __init__(self):
        if ModelUsedAndThreadCount._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.state = ModelUsedAndThreadCountState()
            ModelUsedAndThreadCount._instance = self


class Database:
    def __init__(self, host: str, database: str, user: str, password: str):
        self.host = host
        self.database = database
        self.user = user
        self.password = password

    def get_connection(self):
        # try:
        connection = mysql.connector.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        if connection.is_connected():
            return connection
        # except Error as e:
        #     print(f"Error while connecting to MySQL: {e}")
        #     return None

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
