from flask import Flask, request, render_template, jsonify, redirect, url_for, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
import pytz
import os
import face_recognition
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image, ImageEnhance
from PIL.ExifTags import TAGS, GPSTAGS
import logging
import pickle
from sqlalchemy.sql import text
from apscheduler.schedulers.background import BackgroundScheduler
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['STATIC_FOLDER'] = 'static'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
socketio = SocketIO(app, cors_allowed_origins="*")
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Email configuration for alerts
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'massvikey343@gmail.com'
app.config['MAIL_PASSWORD'] = 'jkzsljwwdivrdqkx'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'doctor', 'admin', 'ddhs'
    email = db.Column(db.String(120), nullable=True)
    facility_id = db.Column(db.Integer, db.ForeignKey('facility.id'), nullable=True)
    predefined_photo_path = db.Column(db.String(200), nullable=True)
    locations = db.relationship('Location', backref='user', lazy=True, foreign_keys='Location.doctor_id')
    attendances = db.relationship('Attendance', backref='user', lazy=True, foreign_keys='Attendance.doctor_id')
    healthcare_services = db.relationship('HealthcareService', backref='user', lazy=True, foreign_keys='HealthcareService.doctor_id')
    alerts = db.relationship('Alert', backref='user', lazy=True, foreign_keys='Alert.doctor_id')

class Division(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    facilities = db.relationship('Facility', backref='division', lazy=True)

class Facility(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    type = db.Column(db.String(20), nullable=False)
    division_id = db.Column(db.Integer, db.ForeignKey('division.id'), nullable=False)
    geofence_id = db.Column(db.Integer, db.ForeignKey('geofence.id'), nullable=False)
    users = db.relationship('User', backref='facility', lazy=True)
    geofence = db.relationship('Geofence', backref='facility', uselist=False)

class Location(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST))
    geofence_status = db.Column(db.String(20), nullable=False, default="Inside")
    phc_area_id = db.Column(db.Integer, nullable=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    photo_status = db.Column(db.String(20), nullable=False)
    photo_path = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST))
    phc_area_id = db.Column(db.Integer, nullable=True)

class Geofence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    lat_min = db.Column(db.Float, nullable=False)
    lat_max = db.Column(db.Float, nullable=False)
    lng_min = db.Column(db.Float, nullable=False)
    lng_max = db.Column(db.Float, nullable=False)

class HealthcareService(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    facility_id = db.Column(db.Integer, nullable=False)
    patient_count = db.Column(db.Integer, nullable=False)
    service_type = db.Column(db.String(50), nullable=False)
    details = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST))
    status = db.Column(db.String(20), nullable=False, default="Completed")
    patient_details_path = db.Column(db.String(200), nullable=True)
    
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(IST))
    status = db.Column(db.String(20), default="Unread")

class KnownFace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, nullable=False)
    encoding = db.Column(db.PickleType, nullable=False)



# Initialize Database with Migration
def init_db():
    with app.app_context():
        logger.info("Initializing database...")
        db.create_all()
        
        # Migration: Add accuracy column to Location table if it doesn't exist
        try:
            result = db.session.execute(text("PRAGMA table_info(location)")).fetchall()
            columns = [row[1] for row in result]
            if 'accuracy' not in columns:
                logger.info("Adding 'accuracy' column to 'location' table...")
                db.session.execute(text("ALTER TABLE location ADD COLUMN accuracy FLOAT"))
                db.session.commit()
                logger.info("Successfully added 'accuracy' column.")
        except Exception as e:
            logger.error(f"Error adding 'accuracy' column: {e}")
            db.session.rollback()
            raise

        # Migration: Add phc_area_id column to Location table if it doesn't exist
        try:
            result = db.session.execute(text("PRAGMA table_info(location)")).fetchall()
            columns = [row[1] for row in result]
            if 'phc_area_id' not in columns:
                logger.info("Adding 'phc_area_id' column to 'location' table...")
                db.session.execute(text("ALTER TABLE location ADD COLUMN phc_area_id INTEGER"))
                db.session.commit()
                logger.info("Successfully added 'phc_area_id' column to 'location' table.")
        except Exception as e:
            logger.error(f"Error adding 'phc_area_id' column to 'location' table: {e}")
            db.session.rollback()
            raise

        # Migration: Add phc_area_id column to Attendance table if it doesn't exist
        try:
            result = db.session.execute(text("PRAGMA table_info(attendance)")).fetchall()
            columns = [row[1] for row in result]
            if 'phc_area_id' not in columns:
                logger.info("Adding 'phc_area_id' column to 'attendance' table...")
                db.session.execute(text("ALTER TABLE attendance ADD COLUMN phc_area_id INTEGER"))
                db.session.commit()
                logger.info("Successfully added 'phc_area_id' column to 'attendance' table.")
        except Exception as e:
            logger.error(f"Error adding 'phc_area_id' column to 'attendance' table: {e}")
            db.session.rollback()
            raise

        # Migration: Add email column to User table if it doesn't exist
        try:
            result = db.session.execute(text("PRAGMA table_info(user)")).fetchall()
            columns = [row[1] for row in result]
            if 'email' not in columns:
                logger.info("Adding 'email' column to 'user' table...")
                db.session.execute(text("ALTER TABLE user ADD COLUMN email VARCHAR(120)"))
                db.session.commit()
                logger.info("Successfully added 'email' column to 'user' table.")
        except Exception as e:
            logger.error(f"Error adding 'email' column to 'user' table: {e}")
            db.session.rollback()
            raise

        # Migration: Add status column to HealthcareService table if it doesn't exist
        try:
            result = db.session.execute(text("PRAGMA table_info(healthcare_service)")).fetchall()
            columns = [row[1] for row in result]
            if 'status' not in columns:
                logger.info("Adding 'status' column to 'healthcare_service' table...")
                db.session.execute(text("ALTER TABLE healthcare_service ADD COLUMN status VARCHAR(20) DEFAULT 'Completed'"))
                db.session.commit()
                logger.info("Successfully added 'status' column to 'healthcare_service' table.")
        except Exception as e:
            logger.error(f"Error adding 'status' column to 'healthcare_service' table: {e}")
            db.session.rollback()
            raise

        # Migration: Add patient_details_path column to HealthcareService table if it doesn't exist
        try:
            result = db.session.execute(text("PRAGMA table_info(healthcare_service)")).fetchall()
            columns = [row[1] for row in result]
            if 'patient_details_path' not in columns:
                logger.info("Adding 'patient_details_path' column to 'healthcare_service' table...")
                db.session.execute(text("ALTER TABLE healthcare_service ADD COLUMN patient_details_path VARCHAR(200)"))
                db.session.commit()
                logger.info("Successfully added 'patient_details_path' column to 'healthcare_service' table.")
        except Exception as e:
            logger.error(f"Error adding 'patient_details_path' column to 'healthcare_service' table: {e}")
            db.session.rollback()
            raise

        
        # Initialize default users with provided email addresses
        if not User.query.first():
            logger.info("No users found, creating default users...")
            doctor_pass_hash = bcrypt.generate_password_hash("pass123").decode('utf-8')
            admin_pass_hash = bcrypt.generate_password_hash("admin123").decode('utf-8')
            ddhs_pass_hash = bcrypt.generate_password_hash("ddhs123").decode('utf-8')
            db.session.add(User(username="doctor1", password=doctor_pass_hash, role="doctor", email="vigneshmcse21@jkkmct.edu.in"))
            db.session.add(User(username="admin1", password=admin_pass_hash, role="admin", email="dhakshinamoorthi1612@gmail.com"))
            db.session.add(User(username="ddhs1", password=ddhs_pass_hash, role="ddhs", email="dhakshinamoorthiecse21@jkkmct.edu.in"))
            db.session.commit()
            logger.info("Default users created with provided emails.")
        else:
            logger.info("Users already exist in the database.")
            doctor1 = User.query.filter_by(username="doctor1").first()
            admin1 = User.query.filter_by(username="admin1").first()
            ddhs1 = User.query.filter_by(username="ddhs1").first()
            if doctor1:
                doctor1.email = "vigneshmcse21@jkkmct.edu.in"
            if admin1:
                admin1.email = "dhakshinamoorthi1612@gmail.com"
            if ddhs1:
                ddhs1.email = "dhakshinamoorthiecse21@jkkmct.edu.in"
            db.session.commit()
            logger.info("Updated existing users with provided email addresses.")
        
        # Initialize divisions
        if not Division.query.first():
            db.session.add(Division(name="Division 1"))
            db.session.commit()
            logger.info("Divisions initialized.")

        # Initialize geofences
        geofences = [
            {
                'name': "JKKMCT PHC",
                'lat_min': 11.50987,
                'lat_max': 11.52987,
                'lng_min': 77.38555,
                'lng_max': 77.40555
            },
            {
                'name': "Singampettai PHC",
                'lat_min': 11.58600174113142 - 0.01,
                'lat_max': 11.58600174113142 + 0.01,
                'lng_min': 77.7275116733288 - 0.01,
                'lng_max': 77.7275116733288 + 0.01
            }
        ]
        
        existing_geofences = Geofence.query.all()
        if len(existing_geofences) != len(geofences):
            for geofence in existing_geofences:
                db.session.delete(geofence)
            for geofence_data in geofences:
                db.session.add(Geofence(**geofence_data))
            db.session.commit()
            logger.info("Geofences initialized or updated with new names.")
        else:
            for i, geofence in enumerate(existing_geofences):
                expected_name = geofences[i]['name']
                if geofence.name != expected_name:
                    geofence.name = expected_name
                    logger.info(f"Updated geofence name from '{geofence.name}' to '{expected_name}'")
            db.session.commit()
            logger.info("Geofence names updated.")

        # Initialize facilities
        if not Facility.query.first():
            division = Division.query.first()
            for geofence in Geofence.query.all():
                facility_type = "PHC"
                db.session.add(Facility(
                    name=geofence.name,
                    type=facility_type,
                    division_id=division.id,
                    geofence_id=geofence.id
                ))
            db.session.commit()
            logger.info("Facilities initialized.")

        # Assign facilities to users
        if User.query.filter_by(facility_id=None).first():
            facility1 = Facility.query.filter_by(name="JKKMCT PHC").first()
            facility2 = Facility.query.filter_by(name="Singampettai PHC").first()
            doctor1 = User.query.filter_by(username="doctor1").first()
            admin1 = User.query.filter_by(username="admin1").first()
            ddhs1 = User.query.filter_by(username="ddhs1").first()
            if doctor1:
                doctor1.facility_id = facility1.id
            if admin1:
                admin1.facility_id = facility1.id
            if ddhs1:
                ddhs1.facility_id = None
            db.session.commit()
            logger.info("Assigned facilities to users.")

# Ensure directories exist
known_faces_dir = os.path.join(os.path.dirname(__file__), 'known_faces')
static_dir = os.path.join(app.root_path, 'static')
for directory in [known_faces_dir, static_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def initialize_known_faces():
    with app.app_context():
        if not KnownFace.query.first():
            known_image_path = os.path.join(known_faces_dir, 'doctor1.jpg')
            if os.path.exists(known_image_path):
                image = preprocess_image(known_image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    db.session.add(KnownFace(doctor_id=1, encoding=pickle.dumps(encodings[0])))
                    db.session.commit()
                    logger.info("Initialized known face for doctor1")
                else:
                    logger.error(f"No face detected in {known_image_path}")
            else:
                logger.error(f"Known face image not found at {known_image_path}")

def preprocess_image(image_path):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    img.save(image_path)
    return face_recognition.load_image_file(image_path)

# Send email alert to multiple recipients with enhanced error logging and HTML support
def send_email_alert(subject, body, to_emails, is_html=False):
    if not isinstance(to_emails, list):
        to_emails = [to_emails]
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = app.config['MAIL_USERNAME']
    msg['To'] = ", ".join(to_emails)

    text_part = MIMEText(body, 'plain')
    msg.attach(text_part)

    if is_html:
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                <h2 style="color: #d9534f; text-align: center;">{subject}</h2>
                <p>{body.replace('\n\n', '</p><p>').replace('\n', '<br>')}</p>
                <hr style="border: 0; border-top: 1px solid #ddd;">
                <p style="font-size: 12px; color: #777; text-align: center;">
                    This is an automated email from the Healthcare Monitoring System. Please do not reply directly to this email.
                </p>
            </div>
        </body>
        </html>
        """
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)

    try:
        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as server:
            server.set_debuglevel(1)
            logger.info(f"Connecting to SMTP server {app.config['MAIL_SERVER']}:{app.config['MAIL_PORT']}")
            server.starttls()
            logger.info(f"Logging in to SMTP server with username {app.config['MAIL_USERNAME']}")
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            logger.info(f"Sending email to {to_emails}")
            server.sendmail(app.config['MAIL_USERNAME'], to_emails, msg.as_string())
            logger.info(f"Email sent successfully to {to_emails}")
    except Exception as e:
        logger.error(f"Failed to send email to {to_emails}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

# Check for absenteeism at 9:30 AM
def check_absenteeism():
    with app.app_context():
        now = datetime.now(IST)
        if now.hour != 10 or now.minute != 1:
            return

        doctors = User.query.filter_by(role="doctor").all()
        for doctor in doctors:
            today = now.date()
            attendance = Attendance.query.filter_by(doctor_id=doctor.id).filter(
                db.func.date(Attendance.timestamp) == today
            ).first()
            if not attendance:
                facility = Facility.query.get(doctor.facility_id)
                subject = "Absenteeism Alert - Attendance Not Marked by 9:30 AM"
                message = (
                    f"Dear DDHS,\n\n"
                    f"**Absenteeism Alert**\n\n"
                    f"This is to inform you that the following doctor has not marked their attendance by 10:01 AM IST today, {today}.\n\n"
                    f"**Details:**\n"
                    f"- Doctor: {doctor.username} (ID: {doctor.id})\n"
                    f"- Facility: {facility.name}\n"
                    f"- Expected Attendance Time: Before 10:01 AM IST\n"
                    f"- Current Status: Attendance not marked\n\n"
                    f"**Action Required:**\n"
                    f"Please follow up with the doctor to ensure compliance with attendance policies. The doctor has been notified to mark their attendance by 12:01 PM to avoid further escalation.\n\n"
                    f"Best regards,\n"
                    f"Healthcare Monitoring System\n"
                    f"Automated Alert Service"
                )
                alert = Alert(doctor_id=doctor.id, message=message)
                db.session.add(alert)
                db.session.commit()
                ddhs = User.query.filter_by(role="ddhs").first()
                if ddhs and ddhs.email:
                    send_email_alert(
                        subject,
                        message,
                        ddhs.email,
                        is_html=True
                    )

# Check for late attendance at 12:01 PM
def check_late_attendance():
    with app.app_context():
        now = datetime.now(IST)
        if now.hour != 9 or now.minute != 30:
            return

        doctors = User.query.filter_by(role="doctor").all()
        for doctor in doctors:
            today = now.date()
            attendance = Attendance.query.filter_by(doctor_id=doctor.id).filter(
                db.func.date(Attendance.timestamp) == today
            ).first()
            if not attendance:
                facility = Facility.query.get(doctor.facility_id)
                subject = "Late Attendance Alert - Immediate Action Required"
                message = (
                    f"Dear {doctor.username},\n\n"
                    f"**Late Attendance Alert**\n\n"
                    f"We have noticed that you have not marked your attendance for today, {today}, at {facility.name} by 09:30 PM IST.\n\n"
                    f"**Details:**\n"
                    f"- Doctor: {doctor.username} (ID: {doctor.id})\n"
                    f"- Facility: {facility.name}\n"
                    f"- Expected Attendance Time: Before 09:30 PM IST\n"
                    f"- Current Status: Attendance not marked\n\n"
                    f"**Action Required:**\n"
                    f"Please mark your attendance immediately to avoid being marked as absent. If you are facing any issues, contact your facility admin ({User.query.filter_by(role='admin', facility_id=doctor.facility_id).first().email}) or the DDHS ({User.query.filter_by(role='ddhs').first().email}) for assistance.\n\n"
                    f"Best regards,\n"
                    f"Healthcare Monitoring System\n"
                    f"Automated Alert Service"
                )
                alert = Alert(doctor_id=doctor.id, message=message)
                db.session.add(alert)
                db.session.commit()
                recipients = []
                if doctor.email:
                    recipients.append(doctor.email)
                admin = User.query.filter_by(role="admin", facility_id=doctor.facility_id).first()
                if admin and admin.email:
                    recipients.append(admin.email)
                ddhs = User.query.filter_by(role="ddhs").first()
                if ddhs and ddhs.email:
                    recipients.append(ddhs.email)
                if recipients:
                    send_email_alert(
                        subject,
                        message,
                        recipients,
                        is_html=True
                    )

# Initialize scheduler for absenteeism and late attendance checks
scheduler = BackgroundScheduler(timezone=IST)
scheduler.add_job(check_absenteeism, 'interval', minutes=1)
scheduler.add_job(check_late_attendance, 'interval', minutes=1)
scheduler.start()

# Run initialization
init_db()
initialize_known_faces()

# EXIF extraction functions
def get_photo_location(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None, None
        gps_info = {}
        for tag, value in exif_data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                for t in value:
                    sub_decoded_tag = GPSTAGS.get(t, t)
                    gps_info[sub_decoded_tag] = value[t]
        if not gps_info:
            return None, None
        def convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info["GPSLatitudeRef"] == "S":
            lat = -lat
        if gps_info["GPSLongitudeRef"] == "W":
            lon = -lon
        return lat, lon
    except Exception as e:
        logger.error(f"Error extracting EXIF data: {e}")
        return None, None

def get_photo_timestamp(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data or 36867 not in exif_data:
            return None
        timestamp_str = exif_data[36867]
        naive_timestamp = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
        return IST.localize(naive_timestamp)
    except Exception as e:
        logger.error(f"Error extracting timestamp: {e}")
        return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            if user.role == 'doctor':
                return redirect(url_for('home'))
            elif user.role == 'admin':
                return redirect(url_for('admin'))
            elif user.role == 'ddhs':
                return redirect(url_for('ddhs'))
        return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    if current_user.role != 'doctor':
        return redirect(url_for('admin' if current_user.role == 'admin' else 'ddhs'))
    geofences = Geofence.query.all()
    if not geofences:
        return "Geofence not configured!", 500
    return render_template('doctor.html', geofences=geofences)

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        return redirect(url_for('home' if current_user.role == 'doctor' else 'ddhs'))
    
    facility = Facility.query.get(current_user.facility_id)
    if not facility:
        return "Admin not assigned to a facility!", 403
    
    doctors = User.query.filter_by(role='doctor', facility_id=facility.id).all()
    doctor_ids = [doctor.id for doctor in doctors]
    
    attendance_records = Attendance.query.filter(Attendance.doctor_id.in_(doctor_ids)).all()
    locations = Location.query.filter(Location.doctor_id.in_(doctor_ids)).all()
    healthcare_services = HealthcareService.query.filter(HealthcareService.doctor_id.in_(doctor_ids)).all()
    
    geofences = Geofence.query.all()
    facilities = Facility.query.all()
    
    for record in attendance_records:
        if record.timestamp.tzinfo is None:
            record.timestamp = IST.localize(record.timestamp)
    for location in locations:
        if location.timestamp.tzinfo is None:
            location.timestamp = IST.localize(location.timestamp)
    if not geofences:
        return "Geofence not configured!", 500
    
    return render_template('admin.html', 
                          attendance_records=attendance_records, 
                          locations=locations,
                          users=doctors,
                          geofences=geofences,
                          healthcare_services=healthcare_services,
                          facilities=facilities)
@app.route('/ddhs')
@login_required
def ddhs():
    if current_user.role != 'ddhs':
        return redirect(url_for('home' if current_user.role == 'doctor' else 'admin'))
    
    divisions = Division.query.all()
    facilities = Facility.query.all()
    attendance_records = Attendance.query.all()
    locations = Location.query.all()
    users = User.query.filter_by(role='doctor').all()
    geofences = Geofence.query.all()
    healthcare_services = HealthcareService.query.all()
    alerts = Alert.query.all()

    # Calculate the most common service type
    service_types = [service.service_type for service in healthcare_services]
    most_common_service = 'N/A'
    if service_types:
        from collections import Counter
        most_common = Counter(service_types).most_common(1)
        most_common_service = most_common[0][0] if most_common else 'N/A'

    for record in attendance_records:
        if record.timestamp.tzinfo is None:
            record.timestamp = IST.localize(record.timestamp)
    for location in locations:
        if location.timestamp.tzinfo is None:
            location.timestamp = IST.localize(location.timestamp)
    for service in healthcare_services:
        if service.timestamp.tzinfo is None:
            service.timestamp = IST.localize(service.timestamp)
    for alert in alerts:
        if alert.timestamp.tzinfo is None:
            alert.timestamp = IST.localize(alert.timestamp)

    return render_template('ddhs.html',
                          divisions=divisions,
                          facilities=facilities,
                          attendance_records=attendance_records,
                          locations=locations,
                          users=users,
                          geofences=geofences,
                          healthcare_services=healthcare_services,
                          alerts=alerts,
                          most_common_service=most_common_service)

@socketio.on('connect', namespace='/admin')
def handle_admin_connect():
    print(f"Admin client connected: {request.sid}")
    facility = Facility.query.get(current_user.facility_id)
    doctors = User.query.filter_by(role='doctor', facility_id=facility.id).all()
    for doctor in doctors:
        last_location = Location.query.filter_by(doctor_id=doctor.id)\
                           .order_by(Location.timestamp.desc()).first()
        if last_location:
            geofence = Geofence.query.get(last_location.phc_area_id) if last_location.phc_area_id else None
            emit('doctor_location_update', {
                'doctor_id': doctor.id,
                'username': doctor.username,
                'latitude': last_location.latitude,
                'longitude': last_location.longitude,
                'accuracy': last_location.accuracy,
                'status': last_location.geofence_status,
                'phc_area': geofence.name if geofence else "Unknown",
                'timestamp': last_location.timestamp.isoformat()
            }, namespace='/admin')

@socketio.on('connect', namespace='/ddhs')
def handle_ddhs_connect():
    print(f"DDHS client connected: {request.sid}")
    doctors = User.query.filter_by(role='doctor').all()
    for doctor in doctors:
        last_location = Location.query.filter_by(doctor_id=doctor.id)\
                           .order_by(Location.timestamp.desc()).first()
        if last_location:
            geofence = Geofence.query.get(last_location.phc_area_id) if last_location.phc_area_id else None
            emit('doctor_location_update', {
                'doctor_id': doctor.id,
                'username': doctor.username,
                'latitude': last_location.latitude,
                'longitude': last_location.longitude,
                'accuracy': last_location.accuracy,
                'status': last_location.geofence_status,
                'phc_area': geofence.name if geofence else "Unknown",
                'timestamp': last_location.timestamp.isoformat()
            }, namespace='/ddhs')

@app.route('/update_location', methods=['POST'])
@login_required
def update_location():
    if current_user.role != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.json
    doctor_id = current_user.id
    phc_area_id = data.get('phc_area_id')
    if not phc_area_id:
        return jsonify({'error': 'PHC area ID required'}), 400
    
    geofence = Geofence.query.get(phc_area_id)
    if not geofence:
        return jsonify({'error': 'Invalid PHC area ID'}), 400

    try:
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        accuracy = float(data.get('accuracy', 0))
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid coordinates'}), 400

    is_inside = (geofence.lat_min <= latitude <= geofence.lat_max and
                 geofence.lng_min <= longitude <= geofence.lng_max)
    status = "Inside" if is_inside else "Outside"

    previous_location = Location.query.filter_by(doctor_id=doctor_id).order_by(Location.timestamp.desc()).first()
    if previous_location and previous_location.geofence_status == "Inside" and status == "Outside":
        facility = Facility.query.get(current_user.facility_id)
        message = f"Doctor {current_user.username} (ID: {doctor_id}) has moved outside the {geofence.name} area at {facility.name} on {datetime.now(IST).date()}."
        alert = Alert(doctor_id=doctor_id, message=message)
        db.session.add(alert)
        db.session.commit()
        recipients = []
        admin = User.query.filter_by(role="admin", facility_id=current_user.facility_id).first()
        if admin and admin.email:
            recipients.append(admin.email)
        ddhs = User.query.filter_by(role="ddhs").first()
        if ddhs and ddhs.email:
            recipients.append(ddhs.email)
        if recipients:
            send_email_alert(
                "Geofence Violation Alert",
                message,
                recipients
            )

    location = Location.query.filter_by(doctor_id=doctor_id).order_by(Location.timestamp.desc()).first()
    if location:
        location.latitude = latitude
        location.longitude = longitude
        location.accuracy = accuracy
        location.geofence_status = status
        location.phc_area_id = phc_area_id
        location.timestamp = datetime.now(IST)
    else:
        location = Location(
            doctor_id=doctor_id,
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            geofence_status=status,
            phc_area_id=phc_area_id
        )
    db.session.add(location)
    db.session.commit()

    socketio.emit('doctor_location_update', {
        'doctor_id': doctor_id,
        'username': current_user.username,
        'latitude': latitude,
        'longitude': longitude,
        'accuracy': accuracy,
        'status': status,
        'phc_area': geofence.name,
        'timestamp': datetime.now(IST).isoformat()
    }, namespace='/admin')
    socketio.emit('doctor_location_update', {
        'doctor_id': doctor_id,
        'username': current_user.username,
        'latitude': latitude,
        'longitude': longitude,
        'accuracy': accuracy,
        'status': status,
        'phc_area': geofence.name,
        'timestamp': datetime.now(IST).isoformat()
    }, namespace='/ddhs')
    logger.debug(f"Emitted doctor_location_update for doctor {doctor_id}: {latitude}, {longitude}, {status}, PHC: {geofence.name}")

    return jsonify({'success': True, 'status': status})

# Update the /report_healthcare_service route to handle patient details file
@app.route('/report_healthcare_service', methods=['POST'])
@login_required
def report_healthcare_service():
    if current_user.role != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        service_type = request.form['service_type']
        patient_count = int(request.form['patient_count'])
        details = request.form.get('details', '')
        status = request.form.get('status', 'Completed')  # Added status field, default to "Completed"
        if status not in ['Completed', 'Ongoing', 'Scheduled']:
            return jsonify({'error': 'Invalid status value'}), 400
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid data'}), 400

    patient_details_path = None
    if 'patient_details' in request.files:
        file = request.files['patient_details']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.pdf')):
            filename = f'patient_details_{current_user.id}_{int(datetime.now(IST).timestamp())}.{file.filename.split(".")[-1]}'
            file_path = os.path.join(static_dir, filename)
            file.save(file_path)
            patient_details_path = filename

    service = HealthcareService(
        doctor_id=current_user.id,
        facility_id=current_user.facility_id,
        patient_count=patient_count,
        service_type=service_type,
        details=details,
        status=status,
        patient_details_path=patient_details_path  # Store the file path
    )
    db.session.add(service)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Healthcare service reported successfully'})

# Add a new route to handle patient details file download
@app.route('/download_patient_details/<int:service_id>')
@login_required
def download_patient_details(service_id):
    if current_user.role not in ['admin', 'ddhs']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    service = HealthcareService.query.get_or_404(service_id)
    if not service.patient_details_path:
        return jsonify({'error': 'No patient details file available'}), 404
    
    file_path = os.path.join(static_dir, service.patient_details_path)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=service.patient_details_path
    )

@app.route('/get_active_doctors')
@login_required
def get_active_doctors():
    if current_user.role not in ['admin', 'ddhs']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    five_min_ago = datetime.now(IST) - timedelta(minutes=5)
    query = db.session.query(
        Location.doctor_id,
        User.username,
        db.func.max(Location.timestamp).label('last_update')
    ).join(User, Location.doctor_id == User.id).filter(
        Location.timestamp >= five_min_ago
    )
    if current_user.role == 'admin':
        facility = Facility.query.get(current_user.facility_id)
        query = query.filter(User.facility_id == facility.id)
    active_doctors = query.group_by(Location.doctor_id, User.username).all()
    
    return jsonify([{
        'doctor_id': d.doctor_id,
        'username': d.username,
        'last_update': d.last_update.isoformat()
    } for d in active_doctors])

@app.route('/get_last_location')
@login_required
def get_last_location():
    if current_user.role not in ['admin', 'ddhs']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    doctor_id = request.args.get('doctor_id')
    if not doctor_id:
        return jsonify({'error': 'Doctor ID required'}), 400
    
    last_location = Location.query.filter_by(doctor_id=doctor_id)\
                       .order_by(Location.timestamp.desc()).first()
    
    if not last_location:
        return jsonify({'error': 'No location data'}), 404
    
    doctor = User.query.get(doctor_id)
    geofence = Geofence.query.get(last_location.phc_area_id) if last_location.phc_area_id else None
    
    return jsonify({
        'doctor_id': doctor_id,
        'username': doctor.username,
        'latitude': last_location.latitude,
        'longitude': last_location.longitude,
        'accuracy': last_location.accuracy,
        'status': last_location.geofence_status,
        'phc_area': geofence.name if geofence else "Unknown",
        'timestamp': last_location.timestamp.isoformat()
    })

@app.route('/upload_photo', methods=['POST'])
@login_required
def upload_photo():
    if current_user.role != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 403
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo captured!'}), 400

    doctor_id = current_user.id
    phc_area_id = request.form.get('phc_area_id')
    if not phc_area_id:
        return jsonify({'error': 'PHC area ID required'}), 400
    
    geofence = Geofence.query.get(phc_area_id)
    if not geofence:
        return jsonify({'error': 'Invalid PHC area ID'}), 400

    now = datetime.now(IST)
    current_time = now.time()
    start_time = datetime.strptime("09:00", "%H:%M").time()
    end_time = datetime.strptime("10:00", "%H:%M").time()
    within_time_window = start_time <= current_time <= end_time

    file = request.files['photo']
    filename = f'doctor_{doctor_id}_{int(datetime.now(IST).timestamp())}.jpg'
    file_path = os.path.join(static_dir, filename)
    file.save(file_path)
    uploaded_image = preprocess_image(file_path)

    photo_lat, photo_lon = get_photo_location(file_path)
    if photo_lat is None or photo_lon is None:
        try:
            latitude = float(request.form.get('live_latitude'))
            longitude = float(request.form.get('live_longitude'))
            accuracy = float(request.form.get('accuracy', 0))
            if accuracy > 1000:
                return jsonify({'error': f'Location accuracy too low! Accuracy: {accuracy} meters.'}), 400
            is_inside_geofence = (geofence.lat_min <= latitude <= geofence.lat_max and
                                  geofence.lng_min <= longitude <= geofence.lng_max)
        except (KeyError, ValueError) as e:
            return jsonify({'error': 'No valid location data provided!'}), 400
    else:
        is_inside_geofence = (geofence.lat_min <= photo_lat <= geofence.lat_max and
                              geofence.lng_min <= photo_lon <= geofence.lng_max)

    geofence_status = "Inside" if is_inside_geofence else "Outside"
    if not is_inside_geofence:
        new_attendance = Attendance(doctor_id=doctor_id, status="Absent", photo_status="Outside PHC", photo_path=filename, phc_area_id=phc_area_id)
        db.session.add(new_attendance)
        db.session.commit()
        return jsonify({'error': f'Photo was taken outside the {geofence.name} area! Marked as Absent.'}), 400

    photo_timestamp = get_photo_timestamp(file_path)
    if photo_timestamp and (datetime.now(IST) - photo_timestamp) > timedelta(minutes=5):
        return jsonify({'error': 'Photo is too old! Please capture a new photo.'}), 400

    uploaded_face_encodings = face_recognition.face_encodings(uploaded_image)
    if not uploaded_face_encodings:
        new_attendance = Attendance(doctor_id=doctor_id, status="Absent", photo_status="No Face Detected", photo_path=filename, phc_area_id=phc_area_id)
        db.session.add(new_attendance)
        db.session.commit()
        return jsonify({'error': 'No face detected in the photo! Marked as Absent.'}), 400

    known_faces = KnownFace.query.filter_by(doctor_id=doctor_id).all()
    if not known_faces:
        return jsonify({'error': 'No known face registered for this doctor! Contact admin to register your face.'}), 400

    known_encodings = [pickle.loads(face.encoding) for face in known_faces]
    logger.debug(f"Number of known encodings for doctor {doctor_id}: {len(known_encodings)}")
    logger.debug(f"Number of uploaded face encodings: {len(uploaded_face_encodings)}")

    best_match = False
    tolerance = 0.5
    for uploaded_encoding in uploaded_face_encodings:
        matches = face_recognition.compare_faces(known_encodings, uploaded_encoding, tolerance=tolerance)
        logger.debug(f"Face comparison matches for doctor {doctor_id}: {matches}")
        if any(matches):
            best_match = True
            break
        distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
        logger.debug(f"Face distances for doctor {doctor_id}: {distances}")
        min_distance = min(distances) if len(distances) > 0 else None
        logger.debug(f"Minimum distance: {min_distance}")

    photo_status = "Face Matched" if best_match else "Face Mismatched"
    status = "Present" if best_match and is_inside_geofence and within_time_window else "Absent"
    if best_match and is_inside_geofence and not within_time_window:
        photo_status += " (Outside Attendance Window)"

    new_attendance = Attendance(doctor_id=doctor_id, status=status, photo_status=photo_status, photo_path=filename, phc_area_id=phc_area_id)
    db.session.add(new_attendance)
    db.session.commit()

    if status == "Absent":
        facility = Facility.query.get(current_user.facility_id)
        subject = "Absenteeism Alert - Doctor Marked Absent"
        message = (
            f"Dear Admin and DDHS,\n\n"
            f"**Absenteeism Alert**\n\n"
            f"This is to inform you that the following doctor has been marked as Absent at {facility.name} on {datetime.now(IST).date()}.\n\n"
            f"**Details:**\n"
            f"- Doctor: {current_user.username} (ID: {doctor_id})\n"
            f"- Facility: {facility.name}\n"
            f"- Reason for Absence: {photo_status}\n"
            f"- Timestamp: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST\n\n"
            f"**Action Required:**\n"
            f"Please investigate the reason for the absence and take appropriate action as per the attendance policy. If needed, contact the doctor at {current_user.email} for clarification.\n\n"
            f"Best regards,\n"
            f"Healthcare Monitoring System\n"
            f"Automated Alert Service"
        )
        alert = Alert(doctor_id=doctor_id, message=message)
        db.session.add(alert)
        db.session.commit()
        recipients = []
        admin = User.query.filter_by(role="admin", facility_id=current_user.facility_id).first()
        if admin and admin.email:
            recipients.append(admin.email)
        ddhs = User.query.filter_by(role="ddhs").first()
        if ddhs and ddhs.email:
            recipients.append(ddhs.email)
        if recipients:
            send_email_alert(
                subject,
                message,
                recipients,
                is_html=True
            )

    if best_match and is_inside_geofence and within_time_window:
        return jsonify({
            'success': True,
            'message': f'Attendance marked as Present: Face Matched inside {geofence.name} area.',
            'geofence_status': geofence_status
        })
    elif best_match and not is_inside_geofence:
        return jsonify({
            'error': f'Attendance marked as Absent: Face Matched but outside {geofence.name} area.',
            'geofence_status': geofence_status
        }), 400
    elif not best_match and is_inside_geofence:
        return jsonify({
            'error': f'Attendance marked as Absent: Face Mismatched inside {geofence.name} area.',
            'geofence_status': geofence_status
        }), 400
    elif best_match and is_inside_geofence and not within_time_window:
        return jsonify({
            'error': f'Attendance marked as Absent: Face Matched inside {geofence.name} area, but outside the 9:00 AM to 10:00 AM attendance window.',
            'geofence_status': geofence_status
        }), 400
    else:
        return jsonify({
            'error': f'Attendance marked as Absent: Face Mismatched outside {geofence.name} area.',
            'geofence_status': geofence_status
        }), 400

@app.route('/admin_upload_doctor_photo', methods=['POST'])
@login_required
def admin_upload_doctor_photo():
    if current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    if 'doctor_id' not in request.form or 'photo' not in request.files:
        return jsonify({'error': 'Doctor ID and photo required!'}), 400
    
    doctor_id = int(request.form['doctor_id'])
    user = User.query.get(doctor_id)
    if not user or user.role != 'doctor':
        return jsonify({'error': 'Invalid Doctor ID!'}), 400
    if user.predefined_photo_path:
        return jsonify({'error': 'Photo already uploaded for this doctor!'}), 400

    file = request.files['photo']
    filename = f'predefined_doctor_{doctor_id}_{int(datetime.now(IST).timestamp())}.jpg'
    file_path = os.path.join(static_dir, filename)
    file.save(file_path)

    user.predefined_photo_path = filename
    db.session.commit()
    logger.info(f"Admin uploaded photo for doctor {doctor_id}")
    return jsonify({'success': True, 'message': f'Photo uploaded for Doctor {doctor_id}'})

@app.route('/generate_report')
@login_required
def generate_report():
    if current_user.role not in ['admin', 'ddhs']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    query = Attendance.query
    if current_user.role == 'admin':
        facility = Facility.query.get(current_user.facility_id)
        doctor_ids = [doctor.id for doctor in User.query.filter_by(role='doctor', facility_id=facility.id).all()]
        query = query.filter(Attendance.doctor_id.in_(doctor_ids))
    
    records = query.all()
    data = []
    for record in records:
        timestamp = record.timestamp
        if timestamp.tzinfo is None:
            timestamp = IST.localize(timestamp)
        geofence = Geofence.query.get(record.phc_area_id) if record.phc_area_id else None
        data.append({
            'Doctor ID': record.doctor_id,
            'Status': record.status,
            'Photo Status': record.photo_status,
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'PHC Area': geofence.name if geofence else 'Unknown'
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance')
    output.seek(0)

    return send_file(
        output,
        download_name='attendance_report.xlsx',
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/generate_healthcare_report')
@login_required
def generate_healthcare_report():
    if current_user.role not in ['admin', 'ddhs']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    query = HealthcareService.query
    if current_user.role == 'admin':
        facility = Facility.query.get(current_user.facility_id)
        doctor_ids = [doctor.id for doctor in User.query.filter_by(role='doctor', facility_id=facility.id).all()]
        query = query.filter(HealthcareService.doctor_id.in_(doctor_ids))
    
    services = query.all()
    data = []
    for service in services:
        timestamp = service.timestamp
        if timestamp.tzinfo is None:
            timestamp = IST.localize(timestamp)
        facility = Facility.query.get(service.facility_id)
        data.append({
            'Doctor ID': service.doctor_id,
            'Facility': facility.name,
            'Service Type': service.service_type,
            'Patients': service.patient_count,
            'Details': service.details or 'N/A',
            'Status': service.status,  # Added status field
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Healthcare Services')
    output.seek(0)

    return send_file(
        output,
        download_name='healthcare_report.xlsx',
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/generate_alerts_report')
@login_required
def generate_alerts_report():
    if current_user.role != 'ddhs':
        return jsonify({'error': 'Unauthorized'}), 403
    
    doctors = User.query.filter_by(role='doctor').all()
    doctor_ids = [doctor.id for doctor in doctors]
    alerts = Alert.query.filter(Alert.doctor_id.in_(doctor_ids)).all()

    data = []
    for alert in alerts:
        timestamp = alert.timestamp
        if timestamp.tzinfo is None:
            timestamp = IST.localize(timestamp)
        data.append({
            'Doctor ID': alert.doctor_id,
            'Message': alert.message,
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Status': alert.status
        })

    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Alerts')
    output.seek(0)

    return send_file(
        output,
        download_name='alerts_report.xlsx',
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/register_face', methods=['POST'])
@login_required
def register_face():
    if current_user.role != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    if 'photo' not in request.files or 'doctor_id' not in request.form:
        return jsonify({'error': 'Photo and doctor_id required!'}), 400
    
    doctor_id = int(request.form['doctor_id'])
    file = request.files['photo']
    file_path = os.path.join(known_faces_dir, f'doctor_{doctor_id}_ref_{int(datetime.now(IST).timestamp())}.jpg')
    file.save(file_path)
    image = preprocess_image(file_path)

    encodings = face_recognition.face_encodings(image)
    if not encodings:
        os.remove(file_path)
        return jsonify({'error': 'No face detected in reference photo!'}), 400

    new_face = KnownFace(doctor_id=doctor_id, encoding=pickle.dumps(encodings[0]))
    db.session.add(new_face)
    db.session.commit()
    logger.info(f"Face registered for Doctor {doctor_id}")
    return jsonify({'success': True, 'message': f'Face registered for Doctor {doctor_id}'})

@app.route('/toggle_alert_status/<int:alert_id>', methods=['POST'])
@login_required
def toggle_alert_status(alert_id):
    logger.debug(f"User {current_user.username} (role: {current_user.role}) attempting to toggle alert {alert_id}")
    if current_user.role != 'ddhs':
        logger.warning(f"Unauthorized access attempt by user {current_user.username}")
        return jsonify({'error': 'Unauthorized'}), 403
    
    alert = Alert.query.get_or_404(alert_id)
    alert.status = "Read" if alert.status == "Unread" else "Unread"
    db.session.commit()
    
    logger.info(f"Alert {alert_id} status updated to {alert.status} by user {current_user.username}")
    return jsonify({
        'success': True,
        'new_status': alert.status,
        'message': f'Alert status updated to {alert.status}'
    })

if __name__ == '__main__':
    socketio.run(app, debug=True)