from flask_sqlalchemy import SQLAlchemy

# Shared SQLAlchemy object used by the Flask app and models.
db = SQLAlchemy()


class Organization(db.Model):
    """Database table for organizations/companies running elections."""

    __tablename__ = "organizations"

    # Internal numeric primary key.
    id = db.Column(db.Integer, primary_key=True)
    # Human-readable company/organization name.
    company_name = db.Column(db.String(255), nullable=False)
    # Smart contract address can be assigned after deployment.
    contract_address = db.Column(db.String(255), nullable=True)
    # Timestamp when organization record was created.
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)


class Voter(db.Model):
    """Database table for registered voters."""

    __tablename__ = "voters"
    # Enforce unique voter email only inside one organization.
    __table_args__ = (db.UniqueConstraint("organization_id", "email", name="uq_voter_org_email"),)

    # Internal numeric primary key.
    id = db.Column(db.Integer, primary_key=True)
    # Human-readable voter name.
    name = db.Column(db.String(255), nullable=False)
    # Hash of voter ID to avoid storing plain voter IDs.
    hashed_voter_id = db.Column(db.String(255), nullable=False)
    # Email must be unique within the same organization.
    email = db.Column(db.String(255), nullable=False)
    # Organization this voter belongs to.
    organization_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=False)
    # Serialized face encoding vector stored as JSON string.
    face_encoding = db.Column(db.Text, nullable=False)
    # Uploaded face image path used for DeepFace image-to-image verification.
    face_image_path = db.Column(db.String(512), nullable=True)
    # Tracks whether this voter has already cast a vote.
    has_voted = db.Column(db.Boolean, default=False, nullable=False)
    # Timestamp when voter record was created.
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)


class Admin(db.Model):
    """Database table for admin users."""

    __tablename__ = "admins"
    # Enforce unique admin email only inside one organization.
    __table_args__ = (db.UniqueConstraint("organization_id", "email", name="uq_admin_org_email"),)

    # Internal numeric primary key.
    id = db.Column(db.Integer, primary_key=True)
    # Admin email must be unique within the same organization.
    email = db.Column(db.String(255), nullable=False)
    # Organization this admin belongs to.
    organization_id = db.Column(db.Integer, db.ForeignKey("organizations.id"), nullable=False)
    # Hashed password generated using werkzeug security helpers.
    password_hash = db.Column(db.String(255), nullable=False)
    # Timestamp when admin record was created.
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
