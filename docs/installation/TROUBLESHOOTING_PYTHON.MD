## Troubleshooting Guide
### Common Issues and Solutions
### 1. Python Not Recognized

Issue:
When running a Python command, you get an error message stating that Python is not recognized as an internal or external command.

Solution:

    Ensure that you have checked the "Add Python to PATH" option during installation.
    Verify that the Python installation directory is included in your system's PATH environment variable.

### 2. Pip Not Found

Issue:
You get an error message stating that pip is not recognized as an internal or external command.

Solution:

    Ensure that pip is installed with Python. You can download the get-pip.py script from the official site and run it to install pip.
    Verify that the Scripts folder within your Python installation directory is included in your system's PATH environment variable.

### 3. Missing Dependencies

Issue:
You encounter errors indicating that certain packages are missing when running the project.

Solution:

    Make sure you have run pip install -r requirements.txt inside the project directory.
    Ensure your virtual environment (if using one) is activated before installing the dependencies.

### 4. Invalid API Key

Issue:
You receive an error stating that the OpenAI API key is invalid or missing.

Solution:

    Double-check that you have added the correct OpenAI API key to the .env file.
    Verify that the .env file is in the correct format: OPENAI_API_KEY=your_api_key_here.

### 5. Permission Denied

Issue:
You encounter a "permission denied" error when running commands.

Solution:

    Ensure you have the necessary permissions to access the project directory and its files.
    On Unix-based systems, you might need to use sudo for administrative privileges.

If you encounter any other issues, please refer to the [official documentation](https://github.com/Googly-Boogly/HappyChoicesAI) or seek help from the community.