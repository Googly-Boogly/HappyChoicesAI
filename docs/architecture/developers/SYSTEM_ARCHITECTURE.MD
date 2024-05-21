## System Architecture

#### Introduction

The system architecture of HappyChoicesAI provides a high-level overview of the system's components, their interactions, and the underlying infrastructure. This document complements the data flow documentation by illustrating the technical framework and flow of data through the system.
High-Level Components

### User Interface (UI)
    Description: The UI is the entry point for users to interact with HappyChoicesAI. It allows users to input dilemmas and view the AI's responses.
    Technologies: Web application (HTML, CSS, JavaScript), API integration.

![UI System Architecture](/docs/img/System_Architcture.jpg)

### API Gateway
    Description: The API Gateway handles communication between the UI and the backend services. It routes user inputs to the appropriate services and returns results to the UI.
    Technologies: RESTful API, GraphQL.

    Endpoint: /process_dilemma
        Inputs:
            dilemma (string, required): The ethical dilemma provided by the user.
            markdown (bool, optional, default: false): Whether to return the output in markdown format.
            save_data (bool, optional, default: false): Whether to save user data for improving HappyChoicesAI in the future.
        Outputs:
            JSON (default) and optionally markdown.

![API Architecture](/docs/img/API_Architecture.jpg)

### Processing Engine

    Description: The core component that performs the historical analysis, key criteria determination, action generation, and thought experiments.
    Sub-components:
        Historical Analysis Module: Searches for parallels between input dilemmas and historical examples.
        Criteria Determination Module: Identifies key criteria for the input dilemma.
        Action Generation Module: Determines potential actions and performs thought experiments.
        Decision Making Module: Weighs all information and selects the best action.
    Technologies: Python, Machine Learning models, NLP libraries (e.g., spaCy, NLTK).

![System Architecture](/docs/img/System_Architcture.jpg)

### State Management

    Description: Manages the state of data throughout the processing pipeline, ensuring that all relevant information is available for decision-making.
    Technologies: In-memory database (e.g., Redis), State management library.

![State Diagram](/docs/img/State.jpg)

### Storage

    Description: Stores historical dilemmas, user inputs, processing results, and final outputs.
    Technologies: SQL/NoSQL databases (e.g., PostgreSQL, MongoDB).

![Database Diagram](/docs/img/database.jpg)

### Output Formatter

    Description: Converts the final results into JSON or markdown format for user download.
    Technologies: Markdown libraries, JSON serialization libraries.

Example Output:

```json
{
    "introduction": "Introduction",
    "insights": "insights",
    "themes": "themes",
    "historical_examples_summary": "example",
    "chosen_best_action_summary": "chosen_best_action_summary",
    "other_thought_experiments_summary": [
      "Example 1",
      "Example 2",
      "Example 3"
    ],
    "lessons_learned": "lessons_learned",
    "conclusion": "conclusion",
    "markdown": "markdown"
}
```




