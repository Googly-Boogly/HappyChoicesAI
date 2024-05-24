## Description for Windows Users:

For Windows, you’ll need to install Docker Desktop and Git. Follow these links and instructions:

Install Docker Desktop: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)

Install Git: [Git for Windows](https://gitforwindows.org/)

Steps for Docker Installation on Windows:

    Download and install Docker Desktop from the link above.

    Follow the installation prompts and start Docker Desktop.

    Ensure Docker is running by opening PowerShell and typing:

```cmd
docker --version
```

Steps for Git Installation on Windows:

    Download and install Git from the link above.

    Follow the installation prompts. You can use the default settings.

    Verify the installation by opening Command Prompt and typing:

```cmd
git --version
```

Steps to Clone the Repository and Run HappyChoicesAI:

    Open PowerShell and navigate to the directory where you want to clone the repository.

    Clone the repository:

```cmd
git clone https://github.com/Googly-Boogly/HappyChoicesAI.git
cd HappyChoicesAI
```

Edit the .env and config.yaml files with your preferred editor.

Run the Docker-Compose command:

```cmd
docker-compose up --build
```

That's it now you can use HappyChoicesAI on Windows!