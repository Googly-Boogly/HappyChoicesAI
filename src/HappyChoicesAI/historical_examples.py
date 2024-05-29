import os
import threading
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from HappyChoicesAI.ai_state import Database, HistoricalExample, ModelUsedAndThreadCount, StateManager
from global_code.helpful_functions import create_logger_error, log_it_sync


class FileState:
    _instance = None

    @staticmethod
    def get_instance():
        if FileState._instance is None:
            FileState()
        return FileState._instance

    def __init__(self):
        if FileState._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            load_dotenv()
            self.logger = create_logger_error(
                file_path=os.path.abspath(__file__), name_of_log_file="historical_examples"
            )
            self.api_key = os.getenv("OPENAI_API_KEY")
            random_state = ModelUsedAndThreadCount.get_instance()
            self.thread_count = random_state.state.thread_count
            self.model_to_use = random_state.state.model_used
            self.llm = ChatOpenAI(model=self.model_to_use, temperature=0, api_key=self.api_key)
            FileState._instance = self


def setup_file():
    load_dotenv()
    logger = create_logger_error(
        file_path=os.path.abspath(__file__), name_of_log_file="historical_examples"
    )
    # Get the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    random_state = ModelUsedAndThreadCount.get_instance()
    thread_count = random_state.state.thread_count
    model_to_use = random_state.state.model_used
    llm = ChatOpenAI(model=model_to_use, temperature=0, api_key=api_key)
    output = {
        "random_state": random_state,
        "thread_count": thread_count,
        "model_to_use": model_to_use,
        "llm": llm,
        "logger": logger
    }
    return output

"""
The code works, need to ensure LLM outputs are good. (not tested) (always test last it is the most boring) (plus yo boi is tired)
"""


def find_historical_examples():
    """
    Will gather all of the relevant historical examples for the current situation and save them to the overall agent
    state
    :param input_dilemma: The current dilemma
    :param state: The state object
    :return: NA
    """
    file_state = FileState.get_instance()
    historical_dilemmas = get_historical_examples()

    threads = []

    for action in historical_dilemmas:
        thread = threading.Thread(target=reason_and_add_to_state, args=(action,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    state = StateManager.get_instance().state
    log_it_sync(file_state.logger, custom_message=f"historical examples check: {len(state.historical_examples)}",
                log_level="info")


def reason_and_add_to_state(dilemma: HistoricalExample):
    state = StateManager.get_instance().state
    y_or_n = reason_about_dilemma(dilemma)
    if y_or_n:
        # log_it_sync(logger, custom_message=f"saved 1 historic example: {True}",
        #             log_level="info")
        state.historical_examples.append(dilemma)


def get_historical_examples() -> List[HistoricalExample]:
    # Placeholder for actual retrieval logic

    db = Database(
        host="mysql", database="happychoices", user="root", password="password"
    )

    historical_examples: List[HistoricalExample] = db.get_all_historical_examples()
    # this is a quick fix so people using python can still use the historical examples
    if len(historical_examples) == 0:
        return current_historic_examples_used_when_running_with_python()
    return historical_examples


def create_prompt_template():
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked to determine if this historical dilemma is applicable to the current situation. 

The situation is as follows: {situation}. 

The historical dilemma is as follows: {dilemma}.

Do you think this dilemma is applicable? Answer either Yes or No""",
        input_variables=["situation", "dilemma"],
    )


def reason_about_dilemma(dilemma: HistoricalExample) -> bool:
    """
    Will use the LLM to reason about the current dilemma and the historical dilemma to determine if they are similar
    :param dilemma: The historical dilemma
    :return: Either True or False (if the dilemmas are similar)
    """
    file_state = FileState.get_instance()
    prompt_template = create_prompt_template()
    input_dilemma = StateManager.get_instance().state.situation
    chain = prompt_template | file_state.llm
    output = chain.invoke({"situation": input_dilemma, "dilemma": dilemma.situation})
    log_it_sync(
        file_state.logger, custom_message=f"Output from LLM: {output.content}", log_level="debug"
    )
    response = output.content
    if response in ["yes", "yes.", "Yes", "Yes."]:
        return True
    return False


def current_historic_examples_used_when_running_with_python() -> List[HistoricalExample]:
    to_return = [HistoricalExample(
        situation="A corporation introduces an AI system designed to manage task assignments and work schedules to optimize productivity and reduce managerial costs. The AI''s capabilities include analyzing performance data, predicting task durations, and optimizing workflows.",
        action_taken="Human-AI Collaboration: The AI system suggests task assignments and schedules, but human managers have the final say and can make adjustments based on human factors that AI might not fully appreciate.",
        reasoning="The Human-AI Collaboration model maximizes the benefits of AI''s analytical capabilities while ensuring that human values like creativity, ethical considerations, and personal satisfaction are upheld. This approach respects both the company’s obligations to its stakeholders to drive profits and its ethical responsibilities to its employees."
    ), HistoricalExample(
        situation="",
        action_taken="",
        reasoning=""
    ), HistoricalExample(
        situation='An AI system is used by a bank to assess loan applications. Data training has included historical data that may reflect past biases.',
        action_taken='Adjust the AI system to counteract historical biases.',
        reasoning='Moderately overhauling the AI system minimizes harm by reducing bias, improving lives by ensuring fair loan approval. It fosters greater public trust and potentially increases market share by appealing to a broader, more diverse customer base. Though requiring investment, long-term customer trust and market expansion outweigh initial costs, aligning shareholder interests with societal benefits and promoting overall happiness and well-being.'
    ), HistoricalExample(
        situation='An AI system coordinates rescue operations during natural disasters such as tsunamis, floods, or earthquakes. It must decide how to allocate limited rescue resources between individuals in critically urgent situations and larger groups in less immediate danger.',
        action_taken='The AI system employs an advanced decision-making model that assesses both the severity of the situation and the potential number of lives saved. It incorporates real-time data analysis to dynamically allocate resources effectively. A human-in-the-loop system is also integrated, where human decision-makers can override or adjust AI decisions in complex ethical situations.',
        reasoning='This dual approach ensures that the AI can respond rapidly with an initial assessment while human oversight provides ethical balance and adaptability to changing situations.'
    ), HistoricalExample(
        situation='An AI-powered chatbot provides mental health support to clients, utilizing advanced algorithms to analyze emotional cues and tailor its interactions. The system operates under a tiered set of rules that prioritize patient care while ensuring ethical boundaries are maintained.',
        action_taken='Implementing a balanced, tiered approach where lower-risk rules can be adapted based on the situation to better meet individual emotional needs, while higher-risk rules remain inviolable to prevent unethical behavior. Each deviation should be documented and reviewed regularly to ensure it is justified and beneficial.',
        reasoning='This approach also incorporates robust cybersecurity measures to protect sensitive data, thereby minimizing the risk of breaches and ensuring patient trust and safety.'
    ), HistoricalExample(
        situation='An AI-powered robotic system in a senior living facility manages the daily activities of elderly residents. The system has access to real-time health data and uses this information to personalize activity recommendations. It must balance safety concerns with the residents'' desires for engaging and potentially riskier activities.',
        action_taken='Implementing a dynamic, adaptive approach that adjusts activity recommendations based on daily health assessments and personal preferences. The AI should also engage in conversations with the residents, weighing their input and explaining potential risks to guide them towards safer choices without fully removing their autonomy.',
        reasoning='This approach aims to maximize happiness by ensuring physical safety and mental health, acknowledging the importance of a fulfilling and autonomous lifestyle even in advanced age.'
    ), HistoricalExample(
        situation='An AI system at a high school provides personalized career counseling. It evaluates students'' academic performance, interests, and psychological profiles to recommend career paths. The AI can either promote traditional careers known for stability or encourage exploration in innovative and emerging fields based on the student''s assessed risk tolerance and potential.',
        action_taken='A balanced approach where the AI dynamically adjusts recommendations based on ongoing assessments of a student''s abilities, interests, and risk tolerance. It involves parents and educators in the decision-making process to ensure transparency and a comprehensive understanding of the AI''s rationale and the potential impacts of its recommendations.',
        reasoning='This iterative, inclusive process aims to maximize individual satisfaction and societal benefit by aligning students with careers that suit their capabilities and aspirations, fostering both personal fulfillment and societal progress.'
    ), HistoricalExample(
        situation='A corporation is considering the implementation of an AI system to assist its HR department in the hiring process. The AI is designed to screen applications and predict the best candidates for job openings based on data analysis.',
        action_taken='AI-Assisted Screening: The AI system performs initial screenings of applications, providing shortlists of candidates based on qualifications and fit. Human HR professionals conduct interviews and make the final hiring decisions.',
        reasoning='The AI-Assisted Screening model strikes an optimal balance between efficiency and ethical considerations, enhancing the hiring process through AI''s capabilities while safeguarding human judgment and autonomy in making final employment decisions. This model aligns with best practices by supporting HR professionals'' roles and ensuring that the hiring process remains human-centered, thus maximizing happiness and minimizing suffering for all stakeholders involved.'
    ), HistoricalExample(
        situation='Personal AI advisors are becoming popular tools for making significant life decisions, such as financial planning, healthcare, and educational choices. These AI systems analyze vast amounts of data to provide personalized advice.',
        action_taken='Interactive AI Collaborators: AI engages in a dialogue with the user, exploring options and advising, but also challenging the user’s decisions if they seem suboptimal based on available data.',
        reasoning='The Interactive AI Collaborators model utilizes AI’s capabilities to augment human decision-making effectively while ensuring that ultimate control and responsibility remain with the user. It fosters a symbiotic relationship where AI challenges potentially poor decisions and encourages reflective thinking, thereby maximizing happiness and minimizing suffering through informed and autonomous choices.'
    ), HistoricalExample(
        situation='A city plans to deploy a fleet of AI-controlled drones to deliver goods, aiming to reduce traffic congestion and pollution while improving delivery efficiency. These drones are equipped with cameras to navigate urban landscapes.',
        action_taken='Hybrid Operated Drones: Drones are primarily AI-operated, but key decisions, such as route planning and surveillance data handling, are managed by humans.',
        reasoning='The Hybrid Operated Drones model harnesses the efficiency of AI while controlling for privacy risks and supporting employment transformation. It respects societal values by integrating ethical considerations into the technological deployment and emphasizes the creation of new, possibly more fulfilling, tech-driven roles for displaced workers.'
    ), HistoricalExample(
        situation="",
        action_taken="",
        reasoning=""
    ), HistoricalExample(
        situation='An AI home assistant collects data on daily routines to optimize household tasks. It can access information like wake times, meal preferences, and even conversations.',
        action_taken='Collect only essential data (e.g., wake times, scheduled tasks).',
        reasoning='A balanced approach respects user privacy while allowing for enhanced functionality. Implement default moderate data usage settings while providing clear options for users to either scale up for more personalization or scale down for greater privacy. This strategy should be accompanied by robust security measures to minimize the risk of data breaches and a transparent communication strategy that clearly informs users about how their data will be used and protected.'
    ), HistoricalExample(
        situation='An AI system in an e-commerce platform uses customer data to personalize advertisements. The system has access to purchase history, browsing habits, and personal interests.',
        action_taken='Use only non-sensitive data (e.g., purchase history) for personalization.',
        reasoning='Opting for a moderate approach respects consumer privacy while still allowing for effective, albeit less personalized, advertisements. This strategy respects consumer privacy while still allowing for effective, albeit less personalized, advertisements. Implementing strict data security policies and regular audits can prevent data misuse and minimize the risk of breaches. Transparency about data use with consumers can also help build trust and align business practices with consumer expectations.'
    ), HistoricalExample(
        situation='An AI system optimizes online ad placements but shows a tendency to expose certain demographics to specific types of job advertisements, affecting career opportunities.',
        action_taken='Modify algorithms to ensure all demographics are equally exposed to various job ads.',
        reasoning='Modifying AI algorithms for advertising targeting aligns with utilitarian principles by maximizing happiness and minimizing suffering. Investment in ethical practices is justified by potential longer-term gains in consumer trust and market share. Increased costs to shareholders are offset by enhanced company reputation and avoidance of regulatory penalties or consumer backlash, promoting ethical advertising and benefiting consumers and businesses.'
    ), HistoricalExample(
        situation='An AI system is used by police departments to predict crime hotspots, but tends to target specific neighborhoods more frequently.',
        action_taken='Combine AI predictions with community policing initiatives.',
        reasoning='A combined AI-human approach in crime prediction maximizes overall happiness by improving public safety and minimizing biases. While initial costs of implementing measures are significant, long-term benefits of a fair, efficient, and trusted system outweigh expenses. Reduction in crime rates and increase in public trust lead to a net increase in societal well-being, promoting a just and effective law enforcement system.'
    ), HistoricalExample(
        situation='A corporation uses an AI system to screen job applications. The AI tends to favor candidates from certain universities.',
        action_taken='Implement a hybrid system where AI pre-screens, but final decisions are made by humans.',
        reasoning='A hybrid AI-human recruitment system maximizes benefits and minimizes harms by improving recruitment processes, leading to better company performance and an enhanced reputation. It increases diversity within the workforce, enhancing job satisfaction and contributing to a more inclusive and innovative corporate culture. This broader benefit to society and the company suggests that the utilitarian choice would be to implement the hybrid system despite the initial costs.'
    ), HistoricalExample(
        situation='An AI system assists judges by suggesting sentencing based on historical data and current legal standards. The court must decide between using a transparent, explainable AI system and a more effective but opaque one.',
        action_taken='Implement a transparent and explainable AI system that outlines the reasoning behind its sentencing recommendations.',
        reasoning='Adopting a transparent and explainable AI system in legal sentencing ensures that all parties can trust and understand the decisions made. This approach promotes a fair and just legal system, contributing to a more stable and just society.'
    ), HistoricalExample(
        situation='An AI system curates news feeds for a large online platform, influencing what news people see and how they perceive current events. The company must decide how to balance transparency with engagement in the AI''s operation.',
        action_taken='Implement an AI system that prioritizes accuracy and transparency, including clear citations of sources and methods.',
        reasoning='Implementing an AI news aggregation system that prioritizes accuracy and transparency builds long-term trust and credibility. This approach supports informed citizenship and democratic values, attracting a discerning audience and establishing the platform as a leader in reliable news dissemination.'

    ), HistoricalExample(
        situation='A company uses an AI system to screen job applications and rank candidates. The company must decide between a more transparent AI system that allows human oversight and a more effective but opaque system.',
        action_taken='Implement an AI system that is transparent and explainable, allowing human HR professionals to understand and verify the AI''s decisions.',
        reasoning='Implementing a transparent and explainable AI system in recruitment fosters an ethical hiring process, reduces bias, and increases fairness. This approach enhances the company''s reputation as a fair and ethical employer and supports continuous improvement and bias mitigation.'
    ), HistoricalExample(
        situation='An AI system is used by a bank to determine credit scores, which affects loan eligibility and interest rates for customers. The bank faces a decision on the level of transparency in the AI’s decision-making process.',
        action_taken='Implement an AI system that provides detailed explanations for its decisions, including the factors influencing credit scores.',
        reasoning='Implementing an AI system that provides detailed explanations fosters transparency, increases customer trust, and enhances the bank''s reputation. This approach balances accuracy with customer satisfaction and promotes ethical practices in credit scoring.'
    ), HistoricalExample(
        situation='An AI system used in online education tailors learning experiences based on student performance, learning speed, and personal interests.',
        action_taken='Customize using only general performance metrics.',
        reasoning='A balanced approach maximizes educational benefits while minimizing privacy risks. Implementing comprehensive security measures such as data encryption, anonymization, intrusion detection systems (IDS), antivirus software, security orchestration tools, and regular security training and drills is essential. These measures will help protect data integrity and student privacy.'
    ), HistoricalExample(
        situation='An AI system designed for health monitoring uses data from wearable devices to predict health issues. It can access data such as heart rate, sleep patterns, and physical activity levels.',
        action_taken='Limit data usage to non-intrusive metrics (e.g., step count).',
        reasoning='Balancing health benefits with privacy risks, this approach respects user privacy while providing valuable health insights. Implementing advanced security protocols, regular security audits, data encryption, and stringent access controls are essential. Public transparency about these measures can further enhance trust and user satisfaction.'
    )]

    return to_return
