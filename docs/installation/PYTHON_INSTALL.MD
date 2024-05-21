## HappyChoicesAI Python Installation Guide

Download python from the official website [here](https://www.python.org/downloads/).

After downloading the installer, run the installer and make sure to check the box that says "Add Python to PATH".

After installing python, open your terminal and run the following command to clone the HappyChoicesAI repository:


```bash
git clone https://github.com/Googly-Boogly/HappyChoicesAI.git
```

Change your current directory to the HappyChoicesAI directory:

```bash
cd HappyChoicesAI
```

Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

Add your OpenAI API key to the .env file located in the project directory.

Execute the following command to run the project:

```bash
python main.py
```

If you wish to modify logging levels or change the model used, you can do so in the config.yaml file. However, it is recommended to use the default settings for the best results.

Use the --output markdown to get a markdown output.

```bash
python main.py --output markdown
```
or a --output json to get a json output.

```bash
python main.py --output json
```

To input the dilemma directly from the command line, use the --input flag.

```bash
python main.py --input "Should I donate to charity?"
```

Enjoy using HappyChoicesAI!
