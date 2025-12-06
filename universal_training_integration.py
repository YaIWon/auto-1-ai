{
    "sms": {
        "enabled": false,
        "twilio_sid": "your_twilio_sid",
        "twilio_token": "your_twilio_token",
        "twilio_from": "+1234567890",
        "phone_numbers": ["+1234567890"]
    },
    "email": {
        "enabled": false,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "password": "your_app_password",
        "recipients": ["your_email@gmail.com"]
    },
    "push": {
        "enabled": false,
        "services": []
    },
    "alarms": {
        "enabled": false,
        "devices": []
    },
    "escalation": {
        "initial_wait": 0,
        "repeat_interval": 600,
        "max_attempts": 100
    }
}
