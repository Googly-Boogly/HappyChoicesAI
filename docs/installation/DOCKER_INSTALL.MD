## HappyChoicesAI Docker Installation Guide

Follow these steps to install and set up HappyChoicesAI:
Prerequisites

Ensure you have Docker installed on your machine. Refer to the official Docker documentation for installation instructions:
[Install Docker](https://docs.docker.com/engine/install/)

Installation Steps

#### Clone the Repository

Open your terminal and run the following command to clone the HappyChoicesAI repository:

```bash
git clone https://github.com/Googly-Boogly/HappyChoicesAI.git
```

#### Navigate to the Project Directory

Change your current directory to the HappyChoicesAI directory:

```bash
cd HappyChoicesAI
```

#### Configure API Key

Add your OpenAI API key to the .env file located in the project directory.

##### Optional Configuration

If you wish to modify logging levels or change the model used, you can do so in the config.yaml file. However, it is recommended to use the default settings for the best results.

#### Build and Run the Docker Container

Execute the following command to build and run the Docker container:


```bash
docker-compose up --build
```