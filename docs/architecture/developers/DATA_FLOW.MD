## Dataflow Documentation for HappyChoicesAI

### Introduction

This document outlines the data flow within HappyChoicesAI, detailing each step from receiving a dilemma to providing a well-reasoned, ethical solution. It includes descriptions of the processes, data handling, and potential diagrams to illustrate the flow.

### Dataflow Steps

### User Input
    Description: The user provides an ethical dilemma through a user interface.
    Data: User-provided dilemma details.



### Historical Dilemmas Analysis
    Description: The AI searches for parallels between the input dilemma and existing historical dilemmas.
    Data: Input dilemma, historical dilemmas.
    Process:
        Compare input dilemma with historical examples.
        Identify and save relevant relationships.

![historic dataflow diagram](/docs/img/historic.jpg)



### Key Criteria Determination
    Description: The AI identifies key criteria for the input dilemma, including individuals, actions, legal issues, consequences, economic impacts, and ethical considerations.
    Data: Input dilemma, key criteria.
    Process:
        Extract key criteria from the dilemma.
        Save criteria to the state.

![key criteria dataflow diagram](/docs/img/key_criteria.jpg)

### Action Determination and Thought Experiments
    Description: The AI determines potential actions and performs thought experiments on each action.
    Data: Key criteria, potential actions.
    Process:
        Determine potential actions.
        Perform thought experiments, including:
            Analyzing parallels using historical examples.
            Assessing changes in key criteria.
            Determining percentage changes for criteria (future ML model enhancement).
            Evaluating impacts on Happiness and Suffering proxies (future ML model enhancement).
        Summarize each thought experiment.

![key criteria dataflow diagram](/docs/img/thought_experiments.jpg)

### Decision-Making
    Description: The AI argues for and against each thought experiment and chooses the best action.
    Data: Thought experiment summaries.
    Process:
        Make arguments for and against each thought experiment.
        Weigh all information and select the best action (future ML model enhancement).

![decision making dataflow diagram](/docs/img/Argue_Pick.jpg)

### Transparency and Summary
    Description: The AI summarizes the process, including thought experiments, common themes, and historical dilemmas used.
    Data: Summaries of thought experiments, chosen action, common themes.
    Process:
        Summarize unchosen thought experiments.
        Create the final summary of the chosen action.
        Identify common themes.
        Provide a short summary of historical dilemmas.

![transparency dataflow diagram](/docs/img/themes_historic.jpg)

### Key Insights and Conclusions
    Description: The AI identifies key insights, creates a final conclusion, and notes lessons learned for future improvements.
    Data: Insights, conclusions, lessons learned.
    Process:
        Determine key insights and trends.
        Create a final conclusion.
        Identify lessons for future improvements.

![transparency dataflow diagram](/docs/img/conclusion_lessons_learned_insights.jpg)

### Output Format
    Description: The user receives the final document in either JSON or markdown format.
    Data: Final document.
    Process:
        Convert the document to the chosen format.
        Provide the document to the user.

![output format dataflow diagram](/docs/img/final_format.jpg)

### Conclusion

This dataflow documentation provides a comprehensive overview of the steps and data handling within HappyChoicesAI. The included diagrams will help visualize the flow and enhance understanding of the process. If there are any specific details you'd like to expand upon or additional diagrams to create, just let me know!