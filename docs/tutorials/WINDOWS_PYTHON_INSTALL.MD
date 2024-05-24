## Windows Documentation for Setting Up HappyChoicesAI

### Step 1: Install Git for Windows

Download Git:
    Visit [Git for Windows](https://gitforwindows.org/) and click on the download link.
    Run the installer once the download is complete.

    Install Git:
        Follow the installation prompts. You can use the default settings.
        Make sure to check the option to add Git to your system PATH.

Verify Installation:

    Open Command Prompt and type:


```cmd
git --version
```

    You should see the installed version of Git.

Step 2: Install Python for Windows

Download Python:
    Visit the python website [Python](https://www.python.org/downloads/windows/) and click on the "Downloads" tab.
    Select the latest version for Windows.

    Install Python:
        Run the installer once the download is complete.
        During installation, make sure to check the box that says "Add Python to PATH".
        Click "Install Now" and follow the installation instructions.

    Verify Installation:

        Open Command Prompt and type:


```cmd
python --version
```
    You should see the installed version of Python.

Step 3: Clone the HappyChoicesAI Repository

    Open Command Prompt:
        Navigate to the directory where you want to clone the repository.

    Clone the Repository:

        Run the following commands:

```cmd
git clone https://github.com/Googly-Boogly/HappyChoicesAI.git
cd HappyChoicesAI
```

Step 4: Edit Configuration Files

    Edit the .env File:
        Open the .env file in a text editor (e.g., Notepad) and add your OpenAI API key.

    Edit the config.yaml File:
        Open the config.yaml file in a text editor and adjust the settings as needed.
        If you have a weaker PC, consider decreasing the thread count.

Step 5: Set Up the Virtual Environment

    Create the Virtual Environment:

        In Command Prompt, navigate to the HappyChoicesAI directory and run:

```cmd
python -m venv venv
```

Activate the Virtual Environment:

    Run the following command:

    
```cmd
venv\Scripts\activate
```

Step 6: Install Dependencies

    Install Required Packages:

        With the virtual environment activated, run:

        
```cmd
pip install -r requirements.txt
```

Step 7: Run the Code

    Execute the Main Script:

        Run the following command:
```cmd
python main.py
```

Input Your Dilemma:

    Follow the prompts to input your dilemma and wait for the results.