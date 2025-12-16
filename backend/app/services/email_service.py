import requests
from app.core.config import settings

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

def send_email(to_email: str, subject: str, html_content: str):
    if not settings.BREVO_API_KEY:
        print("Brevo API Key missing. Email not sent.")
        return False

    headers = {
        "accept": "application/json",
        "api-key": settings.BREVO_API_KEY,
        "content-type": "application/json"
    }
    
    payload = {
        "sender": {"name": settings.SENDER_NAME, "email": settings.SENDER_EMAIL},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content
    }

    try:
        response = requests.post(BREVO_API_URL, json=payload, headers=headers)
        if response.status_code not in [200, 201, 202]:
            print(f"Brevo Error: {response.text}")
            return False
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def send_verification_email(to_email: str, code: str):
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
        <h2>Welcome to Object Detection App</h2>
        <p>Please verify your account using the code below:</p>
        <h1 style="color: #4F46E5; letter-spacing: 5px;">{code}</h1>
        <p>This code expires in 24 hours.</p>
    </div>
    """
    return send_email(to_email, "Verify Your Account", html)

def send_reset_password_email(to_email: str, token: str):
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
        <h2>Password Reset Request</h2>
        <p>You requested to reset your password. Use the token below to reset it:</p>
        <div style="background: #f3f4f6; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 16px;">
            {token}
        </div>
        <p>If you did not request this, please ignore this email.</p>
        <p><small>This token expires in 30 minutes.</small></p>
    </div>
    """
    return send_email(to_email, "Reset Your Password", html)