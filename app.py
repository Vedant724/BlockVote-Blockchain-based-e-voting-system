import hashlib
import json
import os
import random
import uuid
import base64
import requests
from requests import RequestException
from io import BytesIO
from datetime import datetime, timedelta
from functools import wraps
import tempfile

import jwt
import numpy as np
from deepface import DeepFace
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from config import Config
from models import Admin, Organization, Voter, db

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

FACE_MODEL = "SFace"
DETECTOR_BACKEND = "opencv"
# Cosine-distance threshold for same-person decision with SFace embeddings.
FACE_MATCH_THRESHOLD = 0.65
# In-memory OTP store keyed by organization + email.
# {
#   "<org_id>:email@example.com": {
#       "otp": "123456",
#       "expiry": datetime
#   }
# }
OTP_STORE = {}
# Tracks users who have passed OTP verification in current server runtime.
# Key format: "<org_id>:email@example.com"
verified_users = {}
# In-memory candidate list and election status for admin-controlled actions.
# Data is partitioned per organization.
candidates = []
election_status = {}


def _extract_embedding(image_path: str):
    try:
        representations = DeepFace.represent(
            img_path=image_path,
            model_name=FACE_MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )

        if isinstance(representations, dict):
            return representations.get("embedding")

        if isinstance(representations, list) and representations:
            return representations[0].get("embedding")

        return None

    except Exception as e:
        print("Face extraction error:", str(e))
        return None

def _cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 1.0
    similarity = float(np.dot(vec1, vec2) / denom)
    return 1.0 - similarity


def _org_email_key(organization_id: int, email: str) -> str:
    """Build a unique in-memory key for organization-scoped user state."""
    return f"{organization_id}:{email.lower()}"


def _decode_base64_image(face_image_base64: str) -> Image.Image:
    """Decode a raw/data-url base64 image into a normalized RGB Pillow image."""
    encoded_image = str(face_image_base64 or "").strip()
    if "," in encoded_image:
        _, encoded_image = encoded_image.split(",", 1)

    image_bytes = base64.b64decode(encoded_image, validate=True)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _save_face_image(image: Image.Image, destination_path: str) -> None:
    """Persist a face image to disk so DeepFace can compare image-to-image later."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    image.save(destination_path, format="JPEG", quality=95)


def _ensure_voter_face_image_path_column(app: Flask) -> None:
    """Add the face_image_path column for existing SQLite databases if needed."""
    with app.app_context():
        inspector = inspect(db.engine)
        voter_columns = {column["name"] for column in inspector.get_columns("voters")}
        if "face_image_path" not in voter_columns:
            db.session.execute(text("ALTER TABLE voters ADD COLUMN face_image_path VARCHAR(512)"))
            db.session.commit()


def _send_otp_email(sender_email: str, brevo_api_key: str, recipient_email: str, otp: str) -> None:
    """Send OTP email using Brevo transactional email API."""
    api_url = "https://api.brevo.com/v3/smtp/email"

    # Brevo requires API key in `api-key` header and JSON payload.
    headers = {
        "accept": "application/json",
        "api-key": brevo_api_key,
        "content-type": "application/json",
    }

    # Email payload with verified sender and recipient.
    payload = {
        "sender": {"email": sender_email},
        "to": [{"email": recipient_email}],
        "subject": "Your Voting OTP",
        "htmlContent": (
            "<html><body>"
            "<p>Hello,</p>"
            f"<p>Your OTP for secure voting is: <strong>{otp}</strong></p>"
            "<p>This OTP is valid for 5 minutes.</p>"
            "<p>If you did not request this, please ignore this email.</p>"
            "<p>- E-Voting System</p>"
            "</body></html>"
        ),
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=20)
    if response.status_code < 200 or response.status_code >= 300:
        # Raise descriptive error so route handler can log/return safe response.
        raise ValueError(
            f"Brevo API rejected request: {response.status_code} {response.text[:300]}"
        )

def create_app() -> Flask:
    """Application factory to configure and return the Flask app."""
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")

    # Enable CORS for cross-origin frontend/backend communication.
    CORS(app)

    # Attach SQLAlchemy to this Flask app instance.
    db.init_app(app)

    # Create database tables automatically when the backend starts.
    with app.app_context():
        db.create_all()
    _ensure_voter_face_image_path_column(app)

    # Warm up model on startup so first request is faster.
    DeepFace.build_model(FACE_MODEL)

    def _decode_auth_token(required_role: str):
        """Decode JWT and enforce expected role for protected routes."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None, (jsonify({"success": False, "message": "Authorization token is missing"}), 401)

        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return None, (jsonify({"success": False, "message": "Authorization token is missing"}), 401)

        try:
            payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            if payload.get("role") != required_role:
                return None, (
                    jsonify({"success": False, "message": f"{required_role.capitalize()} access required"}),
                    403,
                )
            return payload, None
        except jwt.ExpiredSignatureError:
            return None, (jsonify({"success": False, "message": "Token has expired"}), 401)
        except jwt.InvalidTokenError:
            return None, (jsonify({"success": False, "message": "Invalid token"}), 401)

    def admin_required(route_handler):
        """Allow access only to valid, non-expired JWT tokens with admin role."""

        @wraps(route_handler)
        def wrapped(*args, **kwargs):
            payload, error_response = _decode_auth_token("admin")
            if error_response is not None:
                return error_response

            # Store JWT context for multi-tenant isolation.
            # Protected routes must trust organization_id from token, not client input.
            g.admin_email = payload.get("email")
            g.organization_id = payload.get("organization_id")
            return route_handler(*args, **kwargs)

        return wrapped

    def voter_required(route_handler):
        """Allow access only to valid voter JWT tokens for voter-scoped operations."""

        @wraps(route_handler)
        def wrapped(*args, **kwargs):
            payload, error_response = _decode_auth_token("voter")
            if error_response is not None:
                return error_response

            # Multi-tenant voter context is derived from token claims.
            g.voter_email = payload.get("email")
            g.organization_id = payload.get("organization_id")
            return route_handler(*args, **kwargs)

        return wrapped
    @app.route("/")
    def health() -> str:
        # Basic route to confirm backend service is running.
        return "Backend Running"

    @app.route("/favicon.ico")
    def favicon():
        # Return empty success response so browser favicon checks don't log 404.
        return "", 204

    @app.route("/register", methods=["POST"])
    def register():
        """Register a voter using base64 face image data."""
        payload = request.get_json(silent=True) or request.form

        name = str(payload.get("name", "")).strip()
        voter_id = str(payload.get("voter_id", "")).strip()
        email = str(payload.get("email", "")).strip().lower()
        organization_id_raw = str(payload.get("organization_id", "")).strip()
        face_image_base64 = str(payload.get("face_image", "")).strip()

        # Validate required input fields.
        if not name or not voter_id or not email or not organization_id_raw or not face_image_base64:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "name, voter_id, email, organization_id, and face_image are required",
                    }
                ),
                400,
            )

        try:
            organization_id = int(organization_id_raw)
        except ValueError:
            return jsonify({"success": False, "message": "organization_id must be an integer"}), 400

        # Voter must be registered under a valid organization.
        organization = Organization.query.filter_by(id=organization_id).first()
        if organization is None:
            return jsonify({"success": False, "message": "Organization not found"}), 404

        # Reject duplicate voter registrations by org + email.
        existing_voter = Voter.query.filter_by(email=email, organization_id=organization_id).first()
        if existing_voter is not None:
            return jsonify({"success": False, "message": "Email already exists in this organization"}), 400

        try:
            image = _decode_base64_image(face_image_base64)

            # Convert in-memory image to numpy array for embedding extraction.
            image_array = np.array(image)

            # Generate embedding from image array.
            embedding = _extract_embedding(image_array)
            if not embedding:
                return jsonify({"success": False, "message": "No face detected in image"}), 400

            # Hash voter_id before persistence for privacy/security.
            hashed_voter_id = hashlib.sha256(voter_id.encode("utf-8")).hexdigest()
            image_filename = f"{organization_id}_{secure_filename(email)}_{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            _save_face_image(image, image_path)

            voter = Voter(
                name=name,
                hashed_voter_id=hashed_voter_id,
                email=email,
                organization_id=organization_id,
                face_encoding=json.dumps(embedding),
                face_image_path=image_path,
                has_voted=False,
            )
            db.session.add(voter)
            db.session.commit()

            return jsonify({"success": True, "message": "Voter registered successfully"}), 201
        except (base64.binascii.Error, UnidentifiedImageError):
            db.session.rollback()
            return jsonify({"success": False, "message": "Invalid face_image base64 data"}), 400
        except ValueError:
            # DeepFace raises ValueError when no face is detected.
            db.session.rollback()
            return jsonify({"success": False, "message": "No face detected in image"}), 400
        except Exception:
            db.session.rollback()
            return jsonify({"success": False, "message": "Failed to register voter"}), 500

    @app.route("/verify-face", methods=["POST"])
    def verify_face():
        """Verify a live-captured face image against the stored voter image."""
        payload = request.get_json(silent=True) or request.form

        email = str(payload.get("email", "")).strip().lower()
        organization_id_raw = str(payload.get("organization_id", "")).strip()
        face_image_base64 = str(payload.get("image", payload.get("face_image", ""))).strip()

        if not email or not face_image_base64:
            return jsonify({"success": False, "message": "email and image are required"}), 400

        voter_query = Voter.query.filter_by(email=email)
        if organization_id_raw:
            try:
                organization_id = int(organization_id_raw)
            except ValueError:
                return jsonify({"success": False, "message": "organization_id must be an integer"}), 400
            voter_query = voter_query.filter_by(organization_id=organization_id)

        voter = voter_query.first()
        if voter is None:
            return jsonify({"success": False, "message": "Voter not found"}), 404

        if not voter.face_image_path or not os.path.exists(voter.face_image_path):
            return jsonify({"success": False, "message": "Stored face image not found"}), 404

        temp_file_path = None
        try:
            captured_image = _decode_base64_image(face_image_base64)

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg",
                dir=app.config["UPLOAD_FOLDER"],
            ) as temp_file:
                temp_file_path = temp_file.name
            _save_face_image(captured_image, temp_file_path)

            # DeepFace compares the stored registration image to the live login capture.
            verification = DeepFace.verify(
                img1_path=voter.face_image_path,
                img2_path=temp_file_path,
                model_name=FACE_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )

            if verification.get("distance") is None:
                return jsonify({
                    "success": False,
                    "message": "Face not detected properly. Please align your face."
            })

            verified = bool(verification.get("verified"))

            return jsonify({
                "success": verified,
                "message": "Face verified successfully" if verified else "Face does not match",
                "distance": verification.get("distance"),
                "threshold": verification.get("threshold"),
            })
        except (base64.binascii.Error, UnidentifiedImageError):
            return jsonify({"success": False, "message": "Invalid image data"}), 400
        except Exception:
            return jsonify({"success": False, "message": "Failed to verify face"}), 500
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @app.route("/send-otp", methods=["POST"])
    def send_otp():
        """Generate and store a 6-digit OTP for an existing voter email."""
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        organization_id_raw = str(data.get("organization_id", "")).strip()

        if not email or not organization_id_raw:
            return jsonify({"success": False, "message": "email and organization_id are required"}), 400

        try:
            organization_id = int(organization_id_raw)
        except ValueError:
            return jsonify({"success": False, "message": "organization_id must be an integer"}), 400

        # Only allow OTP generation for registered voters in this organization.
        voter = Voter.query.filter_by(email=email, organization_id=organization_id).first()
        if voter is None:
            return jsonify({"success": False, "message": "Voter not found"}), 404

        # Read Brevo credentials and verified sender email from environment-backed config.
        mail_email = str(app.config.get("BREVO_SENDER_EMAIL", "")).strip() or str(app.config.get("MAIL_EMAIL", "")).strip()
        brevo_api_key = str(app.config.get("BREVO_API_KEY", "")).strip()
        if not mail_email or not brevo_api_key:
            return jsonify({"success": False, "message": "Mail service is not configured"}), 500

        # Create a random 6-digit OTP and set 5-minute expiry.
        otp = f"{random.randint(0, 999999):06d}"
        expiry = datetime.utcnow() + timedelta(minutes=5)

        try:
            # Send OTP to voter's email via Brevo transactional email API.
            _send_otp_email(mail_email, brevo_api_key, email, otp)
        except (RequestException, ValueError) as mail_error:
            # Log exact API/network failure for debugging.
            print(f"[OTP EMAIL ERROR] {type(mail_error).__name__}: {mail_error}")
            return jsonify({"success": False, "message": "Failed to send OTP email"}), 500

        otp_key = _org_email_key(organization_id, email)
        OTP_STORE[otp_key] = {"otp": otp, "expiry": expiry}
        return jsonify({"success": True, "message": "OTP sent successfully"}), 200

    @app.route("/verify-otp", methods=["POST"])
    def verify_otp():
        """Verify OTP for an org-scoped email and remove OTP after success."""
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        otp = str(data.get("otp", "")).strip()
        organization_id_raw = str(data.get("organization_id", "")).strip()

        if not email or not otp or not organization_id_raw:
            return jsonify({"success": False, "message": "email, otp, and organization_id are required"}), 400

        try:
            organization_id = int(organization_id_raw)
        except ValueError:
            return jsonify({"success": False, "message": "organization_id must be an integer"}), 400

        otp_key = _org_email_key(organization_id, email)
        otp_entry = OTP_STORE.get(otp_key)
        if otp_entry is None:
            return jsonify({"success": False, "message": "OTP not found"}), 404

        # Reject expired OTPs and remove them from memory.
        if datetime.utcnow() > otp_entry["expiry"]:
            OTP_STORE.pop(otp_key, None)
            return jsonify({"success": False, "message": "OTP expired"}), 400

        # Reject incorrect OTP values.
        if otp != otp_entry["otp"]:
            return jsonify({"success": False, "message": "Invalid OTP"}), 400

        # OTP is valid; remove it so it cannot be reused.
        OTP_STORE.pop(otp_key, None)
        # Mark this user as OTP-verified for voting permission checks.
        verified_users[otp_key] = True

        # Issue voter JWT after OTP success so voter-scoped endpoints can trust identity.
        expiry_time = datetime.utcnow() + timedelta(hours=1)
        token_payload = {
            "email": email,
            "role": "voter",
            "organization_id": organization_id,
            "exp": expiry_time,
        }
        token = jwt.encode(token_payload, app.config["SECRET_KEY"], algorithm="HS256")

        return (
            jsonify({"success": True, "message": "OTP verified successfully", "token": token}),
            200,
        )

    @app.route("/can-vote", methods=["GET"])
    def can_vote():
        """Return whether a voter is currently allowed to vote."""
        email = request.args.get("email", "").strip().lower()
        organization_id_raw = request.args.get("organization_id", "").strip()
        if not email or not organization_id_raw:
            return jsonify({"can_vote": False}), 400

        try:
            organization_id = int(organization_id_raw)
        except ValueError:
            return jsonify({"can_vote": False}), 400

        # Condition (a): voter must exist in the database for this organization.
        voter = Voter.query.filter_by(email=email, organization_id=organization_id).first()
        if voter is None:
            return jsonify({"can_vote": False}), 200

        # Condition (b): voter must not have voted already.
        if voter.has_voted:
            return jsonify({"can_vote": False}), 200

        # Condition (c): voter must have successfully verified OTP in this organization.
        verify_key = _org_email_key(organization_id, email)
        if not verified_users.get(verify_key):
            return jsonify({"can_vote": False}), 200

        return jsonify({"can_vote": True}), 200

    @app.route("/cast-vote", methods=["POST"])
    def cast_vote():
        """Simulate vote casting by marking has_voted=True in the database."""
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        organization_id_raw = str(data.get("organization_id", "")).strip()
        if not email or not organization_id_raw:
            return jsonify({"success": False, "message": "email and organization_id are required"}), 400

        try:
            organization_id = int(organization_id_raw)
        except ValueError:
            return jsonify({"success": False, "message": "organization_id must be an integer"}), 400

        voter = Voter.query.filter_by(email=email, organization_id=organization_id).first()
        if voter is None:
            return jsonify({"success": False, "message": "Voter not found"}), 404

        # Enforce same conditions as /can-vote before allowing vote cast.
        verify_key = _org_email_key(organization_id, email)
        if voter.has_voted or not verified_users.get(verify_key):
            return jsonify({"success": False, "message": "Voter is not allowed to vote"}), 403

        voter.has_voted = True
        db.session.commit()
        # Optional hardening: remove verified flag once vote is cast.
        verified_users.pop(verify_key, None)

        return jsonify({"success": True, "message": "Vote cast successfully"}), 200

    @app.route("/mark-voted", methods=["POST"])
    @voter_required
    def mark_voted():
        """
        Mark the authenticated voter as voted after blockchain confirmation.

        Multi-tenant isolation:
        organization_id and voter identity are sourced from validated voter JWT claims,
        never from arbitrary frontend input.
        """
        data = request.get_json(silent=True) or {}
        email_input = str(data.get("email", "")).strip().lower()

        try:
            organization_id = g.organization_id
            voter_email = str(g.voter_email or "").strip().lower()
            if organization_id is None or not voter_email:
                return jsonify({"success": False, "message": "Voter scope missing in token"}), 401

            # Prevent a voter token from marking another user as voted.
            if email_input and email_input != voter_email:
                return jsonify({"success": False, "message": "Token/email mismatch"}), 403

            voter = Voter.query.filter_by(email=voter_email, organization_id=organization_id).first()
            if voter is None:
                return jsonify({"success": False, "message": "Voter not found"}), 404

            # Enforce OTP verification gate from current server runtime.
            verify_key = _org_email_key(organization_id, voter_email)
            if not verified_users.get(verify_key):
                return jsonify({"success": False, "message": "OTP verification required"}), 403

            if voter.has_voted:
                return jsonify({"success": True, "message": "Voter already marked as voted"}), 200

            voter.has_voted = True
            db.session.commit()
            verified_users.pop(verify_key, None)
            return jsonify({"success": True, "message": "Voter marked as voted"}), 200
        except Exception:
            db.session.rollback()
            return jsonify({"success": False, "message": "Failed to update voter status"}), 500
    @app.route("/admin-register", methods=["POST"])
    def admin_register():
        """Register a new admin and organization in one flow."""
        data = request.get_json(silent=True) or {}
        company_name = str(data.get("company_name", "")).strip()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", "")).strip()

        if not company_name or not email or not password:
            return jsonify({"success": False, "message": "company_name, email, and password are required"}), 400

        try:
            # Create organization first, then attach admin to that organization.
            organization = Organization(company_name=company_name, contract_address=None)
            db.session.add(organization)
            db.session.flush()

            # Prevent duplicate admin email inside the same organization.
            existing_admin = Admin.query.filter_by(email=email, organization_id=organization.id).first()
            if existing_admin is not None:
                db.session.rollback()
                return jsonify({"success": False, "message": "Admin already exists in this organization"}), 400

            # Hash password before storing in the database.
            password_hash = generate_password_hash(password)
            admin = Admin(email=email, organization_id=organization.id, password_hash=password_hash)
            db.session.add(admin)
            db.session.commit()
        except Exception:
            db.session.rollback()
            return jsonify({"success": False, "message": "Failed to register admin"}), 500

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Admin registered successfully",
                    "organization_id": organization.id,
                }
            ),
            201,
        )

    @app.route("/admin-login", methods=["POST"])
    def admin_login():
        """Authenticate admin credentials and return an org-scoped signed JWT token."""
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", "")).strip()
        organization_id_raw = str(data.get("organization_id", "")).strip()

        if not email or not password or not organization_id_raw:
            return jsonify({"success": False, "message": "email, password, and organization_id are required"}), 400

        try:
            organization_id = int(organization_id_raw)
        except ValueError:
            return jsonify({"success": False, "message": "organization_id must be an integer"}), 400

        admin = Admin.query.filter_by(email=email, organization_id=organization_id).first()
        if admin is None:
            return jsonify({"success": False, "message": "Invalid email, password, or organization"}), 401

        # Compare plain password with stored hash.
        if not check_password_hash(admin.password_hash, password):
            return jsonify({"success": False, "message": "Invalid email, password, or organization"}), 401

        # Create JWT token valid for 1 hour, signed with SECRET_KEY.
        expiry_time = datetime.utcnow() + timedelta(hours=1)
        token_payload = {
            "email": admin.email,
            "role": "admin",
            "organization_id": admin.organization_id,
            "exp": expiry_time,
        }
        token = jwt.encode(token_payload, app.config["SECRET_KEY"], algorithm="HS256")

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Admin login successful",
                    "token": token,
                    "organization_id": admin.organization_id,
                }
            ),
            200,
        )

    @app.route("/admin/candidates/register", methods=["POST"])
    @admin_required
    def register_candidate():
        """Admin-protected candidate registration endpoint (in-memory simulation)."""
        data = request.get_json(silent=True) or {}
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"success": False, "message": "Candidate name is required"}), 400

        # Store candidate under the admin's organization scope.
        candidates.append(
            {
                "name": name,
                "organization_id": g.organization_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        return jsonify({"success": True, "message": "Candidate registered successfully"}), 201

    @app.route("/admin/election/start", methods=["POST"])
    @admin_required
    def start_election():
        """Admin-protected endpoint to start election (simulation)."""
        election_status[g.organization_id] = True
        return jsonify({"success": True, "message": "Election started"}), 200

    @app.route("/admin/election/end", methods=["POST"])
    @admin_required
    def end_election():
        """Admin-protected endpoint to end election (simulation)."""
        election_status[g.organization_id] = False
        return jsonify({"success": True, "message": "Election ended"}), 200

    @app.route("/get-organization", methods=["GET"])
    @admin_required
    def get_organization():
        """
        Return organization details for the authenticated admin tenant only.

        Multi-tenant isolation rule:
        organization_id must come from validated JWT context (g.organization_id),
        never from frontend input, to prevent cross-tenant data access.
        """
        try:
            organization_id = g.organization_id
            if organization_id is None:
                return jsonify({"success": False, "message": "Organization scope missing in token"}), 401

            organization = Organization.query.filter_by(id=organization_id).first()
            if organization is None:
                return jsonify({"success": False, "message": "Organization not found"}), 404

            return (
                jsonify(
                    {
                        "success": True,
                        "id": organization.id,
                        "company_name": organization.company_name,
                        "contract_address": organization.contract_address,
                    }
                ),
                200,
            )
        except Exception:
            return jsonify({"success": False, "message": "Failed to fetch organization"}), 500


    @app.route("/voter-organization", methods=["GET"])
    @voter_required
    def voter_organization():
        """
        Return organization context for authenticated voter only.

        Multi-tenant isolation rule:
        organization_id is sourced from voter JWT (g.organization_id), not from query/body input.
        """
        try:
            organization_id = g.organization_id
            if organization_id is None:
                return jsonify({"success": False, "message": "Organization scope missing in token"}), 401

            organization = Organization.query.filter_by(id=organization_id).first()
            if organization is None:
                return jsonify({"success": False, "message": "Organization not found"}), 404

            return (
                jsonify(
                    {
                        "success": True,
                        "id": organization.id,
                        "company_name": organization.company_name,
                        "contract_address": organization.contract_address,
                    }
                ),
                200,
            )
        except Exception:
            return jsonify({"success": False, "message": "Failed to fetch organization"}), 500

    @app.route("/save-contract-address", methods=["POST"])
    @admin_required
    def save_contract_address():
        """
        Save smart contract address for the authenticated admin's organization.

        Multi-tenant isolation rule:
        organization_id is taken from JWT context (g.organization_id), not from client payload.
        This prevents one tenant from modifying another tenant's contract settings.
        """
        data = request.get_json(silent=True) or {}
        contract_address = str(data.get("contract_address", "")).strip()

        if not contract_address:
            return jsonify({"success": False, "message": "contract_address is required"}), 400

        # Basic Ethereum address format validation.
        if not (contract_address.startswith("0x") and len(contract_address) == 42):
            return jsonify({"success": False, "message": "Invalid contract address format"}), 400

        try:
            organization_id = g.organization_id
            if organization_id is None:
                return jsonify({"success": False, "message": "Organization scope missing in token"}), 401

            organization = Organization.query.filter_by(id=organization_id).first()
            if organization is None:
                return jsonify({"success": False, "message": "Organization not found"}), 404

            organization.contract_address = contract_address
            db.session.commit()

            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Contract address saved successfully",
                        "contract_address": organization.contract_address,
                    }
                ),
                200,
            )
        except Exception:
            db.session.rollback()
            return jsonify({"success": False, "message": "Failed to save contract address"}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)






