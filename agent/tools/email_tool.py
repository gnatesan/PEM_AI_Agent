"""
LangChain BaseTool: EmailTool
Allows the agent to send emails to clinicians, patients, or hospital staff.
"""

import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional
from langchain.tools import BaseTool

class EmailTool(BaseTool):
    """LangChain-compatible tool for sending emails."""

    name: str = "email_tool"
    description: str = (
        "Send an email message to a patient, clinician, or hospital staff "
        "with a provided subject and body."
    )

    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: Optional[str] = None
    sender_password: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load from environment variables if available
        self.sender_email = os.getenv("EMAIL_ADDRESS")
        self.sender_password = os.getenv("EMAIL_PASSWORD")

        if not self.sender_email or not self.sender_password:
            print(" Warning: EMAIL_ADDRESS or EMAIL_PASSWORD not set in .env")

    def _run(
        self,
        recipient_email: str,
        subject: str,
        message: str,
        run_manager: Optional[object] = None,
    ) -> str:
        """Synchronously send an email."""
        try:
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = recipient_email

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            return f"Email successfully sent to {recipient_email} with subject '{subject}'."

        except Exception as e:
            return f"Failed to send email: {str(e)}"

    async def _arun(
        self,
        recipient_email: str,
        subject: str,
        message: str,
        run_manager: Optional[object] = None,
    ) -> str:
        """Async version of email sending."""
        return self._run(recipient_email, subject, message)

