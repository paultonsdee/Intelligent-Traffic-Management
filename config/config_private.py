import os

def load_env_variables(env_file=".env.private"):
    """Load environment variables from the given .env file."""
    if not os.path.exists(env_file):
        return {}

    env_variables = {}
    with open(env_file, "r") as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                env_variables[key] = value
    return env_variables

def prompt_for_missing_variables(existing_vars):
    """Prompt the user for missing environment variables."""
    required_keys = ["FROM_EMAIL", "EMAIL_PASSWORD", "TO_EMAIL", "GEMINI_API_KEY"]
    help_text = {
        "FROM_EMAIL": "Email address to send emails from",
        "EMAIL_PASSWORD": "Application password for the email account (https://knowledge.workspace.google.com/kb/how-to-create-app-passwords-000009237)",
        "TO_EMAIL": "Email address to send emails to",
        "GEMINI_API_KEY": "API key for the Gemini model (https://aistudio.google.com/apikey)"
    }

    config = {}

    for key in required_keys:
        if key in existing_vars and existing_vars[key]:
            print(f"{key} is already set.")
            config[key] = existing_vars[key]
        else:
            config[key] = input(f"{help_text[key]}: ").strip()

    return config

def write_to_env_private(config, env_file=".env.private"):
    """Write configuration details to the .env.private file."""
    with open(env_file, "w") as private_file:
        for key, value in config.items():
            private_file.write(f"{key}={value}\n")
    print(f"Configuration saved to {env_file}.")

def main():
    env_file = ".env.private"

    # Load existing variables
    existing_vars = load_env_variables(env_file)

    # Prompt for missing variables
    config = prompt_for_missing_variables(existing_vars)

    # Write updated configuration back to the .env.private file
    write_to_env_private(config, env_file)

if __name__ == "__main__":
    main()
