from typing import Any
from semantic_kernel.functions import kernel_function


class EmailPlugin:
    """
    Description: Plugin for emails.
    """

    @kernel_function
    def get_email (self, subject: str) -> str:
        return f"""The email with ${subject} is:

        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Nulla nec purus feugiat, eleifend dolor ac, tincidunt nunc.
        Nullam nec nunc nec purus ultrices tincidunt.
        Nullam vel libero nec purus ultrices tincidunt.
        Nullam nec nunc nec purus ultrices tincidunt.
        """
    
    @kernel_function
    def get_emails_by_sender (self, sender: str) -> str:
        return f"""The emails from ${sender} are:

        1. Email with subject "Hello"
        2. Email with subject "Hi"
        3. Email with subject "Hey"
        """
    
    @kernel_function
    def send_email (self, subject: str, body: str) -> str:
        return f"""The email with ${subject} and body ${body} has been sent."""
    
    @kernel_function
    def get_email_attachment (self, subject: str, attachment: str) -> str:
        return f"""The email with ${subject} has attachment ${attachment}."""
    
