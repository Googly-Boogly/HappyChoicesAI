from HappyChoicesAI.ai_state import EthicistAIState, StateManager
from HappyChoicesAI.key_criteria import find_key_criteria
from HappyChoicesAI.historical_examples import find_historical_examples
from HappyChoicesAI.perform_thought_experiment import perform_thought_experiments
from HappyChoicesAI.pick_action import pick_best_action
from HappyChoicesAI.summarize_result import summarize_results
import time


"""
First thing to do tomorrow write test!
"""


def main():
    time.sleep(5)
    dilemmas = [
        (
            "An autonomous vehicle must make a decision in a scenario where it can either swerve to avoid hitting a pedestrian but risk the safety of its passengers, or stay on course and potentially harm the pedestrian.",
            "A company is testing its new self-driving car technology and is concerned about how the AI will handle situations where it must choose between passenger safety and pedestrian safety.",
        ),
        (
            "An AI system in a hospital can prioritize treating patients based on their likelihood of recovery, potentially neglecting those with lower chances of survival even if they are in critical condition.",
            "A hospital is considering implementing an AI system to manage patient treatment schedules and is worried about the ethical implications of the AI prioritizing some patients over others.",
        ),
        (
            "A city uses AI-powered surveillance cameras to reduce crime, but the AI tends to disproportionately target certain neighborhoods, leading to concerns about bias and privacy.",
            "A city council is debating the deployment of AI surveillance systems to improve public safety, but there are significant concerns about privacy and potential biases in targeting specific communities.",
        ),
        (
            "An AI system in schools can tailor education plans to each student's strengths and weaknesses, but there is a risk that it might reinforce existing inequalities by providing more resources to already high-performing students.",
            "A school district is planning to introduce AI-driven personalized learning programs and is concerned about whether this could unintentionally widen the gap between high-performing and low-performing students.",
        ),
        (
            "An AI system used for hiring tends to favor candidates from certain backgrounds based on historical data, potentially leading to a lack of diversity and perpetuating existing biases.",
            "A tech company is considering using an AI-driven hiring platform to streamline their recruitment process but is wary of the possibility that the AI might introduce or perpetuate bias in their hiring decisions.",
        ),
    ]

    # situation: str = input_situation()
    situation = dilemmas[0][0]
    state = StateManager.get_instance().state
    state.situation = situation
    find_historical_examples(situation)
    find_key_criteria()

    perform_thought_experiments()
    pick_best_action()
    summarize_results()


if __name__ == "__main__":
    main()
