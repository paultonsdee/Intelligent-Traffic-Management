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
        self.email_cooldown = 20  # Increased cooldown to 60 seconds

        # Email settings
        self.from_email = Config.FROM_EMAIL
        self.from_email_nickname = Config.FROM_EMAIL_NICKNAME
        self.password = Config.EMAIL_PASSWORD
        self.to_email = Config.TO_EMAIL

        # Configure logging for this class
        self.logger = logging.getLogger(self.__class__.__name__)

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
        self.logger.info("Current Violations:")
        self.logger.info("-" * 50)
        self.logger.info(f"Large Vehicles in Motorcycle Lane: {self.violations['large_vehicles']}")
        self.logger.info(f"Motorcycles in Large Vehicle Lane: {self.violations['motorcycles']}")
        self.logger.info(f"Illegal Parked Vehicles: {self.violations['illegal_parking']}")
        self.logger.info(f"Red Light Running Vehicles: {self.violations['red_light']}")
        self.logger.info("-" * 50)

    def send_violation_email(self, violation_type, count):
        try:
            message = MIMEMultipart()
            message["From"] = f"{self.from_email_nickname} <{self.from_email}>"
            message["To"] = self.to_email
            message["Subject"] = "Violation Alert"

            body = "Traffic violations have exceeded the threshold:\n\n"
            body += f"Type: {violation_type.replace('_', ' ').title()}\nCount: {count}\n\n"

            message.attach(MIMEText(body, "plain"))

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

        # Check if any violations exceed thresholds
        if violations_to_alert:
            for vt, cnt in violations_to_alert.items():
                self.send_violation_email(vt, cnt)
            self.last_email_time = current_time

        # Check if cooldown has passed
        elif violations_detected and (current_time - self.last_email_time >= self.email_cooldown):
            # Reset violation counts after email sent
            violations_detected = {vt: cnt for vt, cnt in self.violations.items() if cnt != 0}
            for vt, cnt in violations_detected.items():
                self.send_violation_email(vt, cnt)
            self.last_email_time = current_time
            