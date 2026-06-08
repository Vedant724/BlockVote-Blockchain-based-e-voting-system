import os

from dotenv import load_dotenv

# Always load the .env that lives next to this config file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    # SQLite database file required by the project specification.
    SQLALCHEMY_DATABASE_URI = "sqlite:///database.db"
    # Disable tracking to reduce overhead and warnings.
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # SECRET_KEY is used to sign JWT tokens.
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    # Verified sender email for Brevo transactional OTP emails.
    MAIL_EMAIL = os.getenv("MAIL_EMAIL", "")
    # Brevo API key used for transactional email API authentication.
    BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
    # Optional explicit Brevo sender email; falls back to MAIL_EMAIL.
    BREVO_SENDER_EMAIL = os.getenv("BREVO_SENDER_EMAIL", "")
