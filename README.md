# HappyChoicesAI

HappyChoicesAI is an AI-driven utilitarian ethicist agent designed to help users make ethical decisions. By analyzing user-inputted dilemmas, HappyChoicesAI suggests the most ethical actions based on utilitarian principles, aiming to maximize happiness and minimize suffering. This project leverages advanced AI technologies to process and evaluate ethical decisions, providing grounded and pragmatic solutions to complex dilemmas.

## Installation

### Docker Installation

1. **Install Docker**:
    - On Windows, download and install Docker Desktop from [Docker's official website](https://www.docker.com/products/docker-desktop).
    - On Ubuntu:
      ```bash
      sudo apt update
      sudo apt install docker.io
      sudo systemctl start docker
      sudo systemctl enable docker
      sudo usermod -aG docker $USER
      ```

2. **Clone the Repository**:
    ```bash
    git clone https://github.com/Googly-Boogly/HappyChoicesAI.git
    cd HappyChoicesAI
    ```

3. **Configure the Environment**:
    - Edit the `.env` file and `config.yaml` file:
      ```bash
      nano .env
      nano config.yaml
      ```

4. **Run the Docker Container**:
    ```bash
    docker-compose up --build
    ```

### Python Installation

1. **Install Python**:
    - On Windows, download and install Python from [Python's official website](https://www.python.org/).
    - On Ubuntu:
      ```bash
      sudo apt update
      sudo apt install python3 python3-venv python3-pip
      ```

2. **Clone the Repository**:
    ```bash
    git clone https://github.com/Googly-Boogly/HappyChoicesAI.git
    cd HappyChoicesAI
    ```

3. **Configure the Environment**:
    - Edit the `.env` file and `config.yaml` file:
      ```bash
      nano .env
      nano config.yaml
      ```

4. **Set Up the Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

5. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

6. **Run the Python Program**:
    ```bash
    python main.py
    ```

### Demonstration Video

Watch a sped-up demonstration of HappyChoicesAI in action, where we input a dilemma and get an output in markdown format.

[Link to demonstration video]

## Technologies Used

- Docker
- Python
- MySQL
- Langchain

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Email: [HappyChoicesAI@proton.me](mailto:HappyChoicesAI@proton.me)  
Project Link: [https://github.com/Googly-Boogly/HappyChoicesAI](https://github.com/Googly-Boogly/HappyChoicesAI)

## AI-Enhanced

Special Thanks ChatGPT!
