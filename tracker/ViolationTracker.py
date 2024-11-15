# --------------------- Violation Tracker --------------------- #

import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from config.Config import Config
from config.config_logger import logger
import os

class ViolationTracker:
    """
    Tracks traffic violations and sends email alerts when violations exceed thresholds.
    """
    def __init__(self):
        self.violations = {
            'large_vehicles': 0,    # Region 1 (classes 1,2,3)
            'motorcycles': 0,       # Region 2 (class 0)
            'illegal_parking': 0,   # Region 3 (v=0)
            'red_light': 0          # Outside Region 3
        }
        self.thresholds = Config.VIOLATION_THRESHOLDS
        self.last_email_time = 0
        self.email_cooldown = Config.SEND_EACH

        # Email settings
        self.from_email = Config.FROM_EMAIL
        self.from_email_nickname = Config.FROM_EMAIL_NICKNAME
        self.password = Config.EMAIL_PASSWORD
        self.to_email = Config.TO_EMAIL

        # Configure logging for this class
        self.logger = logger.getChild(self.__class__.__name__)

        # Lock for thread safety
        self.lock = threading.Lock()

    def get_violation_count(self, violation_type):
        with self.lock:
            return self.violations.get(violation_type, 0)

    def update_violations(self, violation_type, count):
        with self.lock:
            self.violations[violation_type] = count
            self.print_violations()
            self.check_and_send_email()

    def print_violations(self):
        # Clear terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Current Violations:")
        print("-" * 50)
        print(f"Large Vehicles in Motorcycle Lane: {self.violations['large_vehicles']}")
        print(f"Motorcycles in Large Vehicle Lane: {self.violations['motorcycles']}")
        print(f"Illegal Parked Vehicles: {self.violations['illegal_parking']}")
        print(f"Red Light Running Vehicles: {self.violations['red_light']}")
        print("-" * 50)

    def send_violation_email(self, violation_type, count):
        try:
            message = MIMEMultipart()
            message["From"] = f"{self.from_email_nickname} <{self.from_email}>"
            message["To"] = self.to_email
            message["Subject"] = Config.EMAIL_SUBJECT

            violation_messages = {
                'large_vehicles': "There are <b><span style='color: #12ffe7;'>{count}</span></b> large vehicle(s) <b><span style='color: #34f7e4;'>encroaching</span></b> the lane designated for motorcycles.",
                'motorcycles': "There are <b><span style='color: #12ffe7;'>{count}</span></b> motorcycle(s) <b><span style='color: #34f7e4;'>encroaching</span></b> the lane designated for large vehicles.",
                'illegal_parking': "There are <b><span style='color: #12ffe7;'>{count}</span></b> <b><span style='color: #34f7e4;'>illegally</span></b> parked vehicle(s) in the restricted area.",
                'red_light': "There are <b><span style='color: #12ffe7;'>{count}</span></b> vehicle(s) <b><span style='color: #34f7e4;'>running a red light</span></b>."
            }
            body = violation_messages.get(violation_type, "There is a traffic violation.")
            body = body.format(count=count)

            message.attach(MIMEText(body, "html"))

            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.from_email, self.password)
            server.send_message(message)
            server.quit()

            self.logger.info(f"Sent email for violations: {violation_type}")

            # Reset violation counts after email sent
            self.violations[violation_type] = 0

        except Exception as e:
            self.logger.error(f"Failed to send email for {violation_type}: {e}")

    def check_and_send_email(self):
        current_time = time.time()
        violations_to_alert = {
            vt: cnt for vt, cnt in self.violations.items()
            if cnt >= self.thresholds.get(vt, 10)
        }
        violations_detected = {vt: cnt for vt, cnt in self.violations.items() if cnt != 0}

        # Check if any violations exceed thresholds
        if violations_to_alert:
            for vt, cnt in violations_to_alert.items():
                self.send_violation_email(vt, cnt)
            self.last_email_time = current_time

        # Check if cooldown has passed
        elif violations_detected and (current_time - self.last_email_time >= self.email_cooldown):
            # Reset violation counts after email sent
            for vt, cnt in violations_detected.items():
                self.send_violation_email(vt, cnt)
            self.last_email_time = current_time
