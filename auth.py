import os
import json
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, request, current_app
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

# Get the absolute path of the directory where this script resides
_AUTH_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE_PATH = os.path.join(_AUTH_DIR, '.auth_key')
USERS_FILE_PATH = os.path.join(_AUTH_DIR, 'users.json')

# Load environment variables
load_dotenv()

# Generate or load encryption key for additional data protection
def get_or_create_key():
    """Create or load the Fernet key from .auth_key using an absolute path."""
    try:
        with open(KEY_FILE_PATH, 'rb') as key_file:
            key = key_file.read()
    except FileNotFoundError:
        key = Fernet.generate_key()
        with open(KEY_FILE_PATH, 'wb') as key_file:
            key_file.write(key)
        # Set restrictive permissions
        os.chmod(KEY_FILE_PATH, 0o600)
    return key

# Initialize encryption
CIPHER_SUITE = Fernet(get_or_create_key())

class User(UserMixin):
    def __init__(self, username, password_hash=None, role='user', is_active=True, created_at=None, last_login=None):
        self.id = username  # Use username as ID
        self.username = username
        self.password_hash = password_hash
        self.role = role  # 'superadmin' or 'user'
        self._is_active = is_active  # Use private attribute to avoid conflict with UserMixin
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.last_login = last_login
        self.failed_attempts = 0
        self.locked_until = None

    @property
    def is_active(self):
        """Override UserMixin's is_active property"""
        return self._is_active

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_superadmin(self):
        return self.role == 'superadmin'
    
    def to_dict(self):
        return {
            'username': self.username,
            'password_hash': self.password_hash,
            'role': self.role,
            'is_active': self._is_active,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'failed_attempts': self.failed_attempts,
            'locked_until': self.locked_until
        }
    
    @classmethod
    def from_dict(cls, data):
        user = cls(
            username=data['username'],
            password_hash=data['password_hash'],
            role=data.get('role', 'user'),
            is_active=data.get('is_active', True),
            created_at=data.get('created_at'),
            last_login=data.get('last_login')
        )
        user.failed_attempts = data.get('failed_attempts', 0)
        user.locked_until = data.get('locked_until')
        return user

class AuthManager:
    def __init__(self, app=None):
        self.users_file = USERS_FILE_PATH  # Use absolute path
        self.login_manager = LoginManager()
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        self.app = app
        self.login_manager.init_app(app)
        self.login_manager.login_view = 'login'
        self.login_manager.login_message = 'Please log in to access this page.'
        self.login_manager.session_protection = 'strong'
        
        # Session configuration for security
        app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'  # HTTPS only in production
        app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
        app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
        app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)  # Session timeout
        
        @self.login_manager.user_loader
        def load_user(username):
            return self.get_user(username)
        
        # Ensure users are initialized on startup
        self.check_and_initialize_users()
    
    def check_and_initialize_users(self):
        """Check if the users file exists and initialize if not."""
        if not os.path.exists(self.users_file):
            print("Users file not found. Initializing default users...")
            self.initialize_default_users()
    
    def initialize_default_users(self):
        """Initialize users from environment variables"""
        users = {}
        
        # Get superadmin credentials from environment
        superadmin_username = os.environ.get('SUPERADMIN_USERNAME', 'admin')
        superadmin_password = os.environ.get('SUPERADMIN_PASSWORD', secrets.token_urlsafe(16))
        
        # Create superadmin user
        users[superadmin_username] = User(
            username=superadmin_username,
            password_hash=generate_password_hash(superadmin_password, method='pbkdf2:sha256', salt_length=16),
            role='superadmin'
        ).to_dict()
        
        # Check for additional pre-configured users
        for i in range(1, 10):  # Support up to 9 additional users
            username = os.environ.get(f'USER{i}_USERNAME')
            password = os.environ.get(f'USER{i}_PASSWORD')
            if username and password:
                users[username] = User(
                    username=username,
                    password_hash=generate_password_hash(password, method='pbkdf2:sha256', salt_length=16),
                    role='user'
                ).to_dict()
        
        # Save encrypted users data
        self.save_users(users)
        
        # Print default credentials on first run (only if generated)
        if superadmin_password == os.environ.get('SUPERADMIN_PASSWORD'):
            print(f"\n{'='*50}")
            print("Authentication System Initialized")
            print(f"Superadmin Username: {superadmin_username}")
            print(f"Superadmin Password: {superadmin_password}")
            print("Please save these credentials and set them as environment variables!")
            print(f"{'='*50}\n")
    
    def load_users(self):
        """Load and decrypt users data"""
        if not os.path.exists(self.users_file):
            return {}
        
        try:
            with open(self.users_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = CIPHER_SUITE.decrypt(encrypted_data)
            users_data = json.loads(decrypted_data.decode())
            
            # Convert dict data back to User objects
            users = {}
            for username, user_data in users_data.items():
                users[username] = User.from_dict(user_data)
            
            return users
        except Exception as e:
            print(f"Error loading users: {e}")
            return {}
    
    def save_users(self, users):
        """Encrypt and save users data"""
        # Convert User objects to dict for serialization
        users_data = {}
        for username, user in users.items():
            if isinstance(user, User):
                users_data[username] = user.to_dict()
            else:
                users_data[username] = user
        
        # Encrypt the data
        json_data = json.dumps(users_data).encode()
        encrypted_data = CIPHER_SUITE.encrypt(json_data)
        
        # Save with restrictive permissions
        with open(self.users_file, 'wb') as f:
            f.write(encrypted_data)
        os.chmod(self.users_file, 0o600)
    
    def get_user(self, username):
        """Get a user by username"""
        users = self.load_users()
        user_data = users.get(username)
        if user_data and isinstance(user_data, dict):
            return User.from_dict(user_data)
        return user_data
    
    def authenticate_user(self, username, password):
        """Authenticate a user with rate limiting and account locking"""
        users = self.load_users()
        user = users.get(username)
        
        if not user:
            # Don't reveal if username exists
            return None, "Invalid username or password"
        
        # Check if account is locked
        if user.locked_until:
            locked_until = datetime.fromisoformat(user.locked_until)
            if datetime.utcnow() < locked_until:
                remaining = (locked_until - datetime.utcnow()).seconds // 60
                return None, f"Account locked. Try again in {remaining} minutes."
            else:
                # Unlock the account
                user.locked_until = None
                user.failed_attempts = 0
        
        # Check password
        if user.check_password(password):
            # Reset failed attempts on successful login
            user.failed_attempts = 0
            user.last_login = datetime.utcnow().isoformat()
            
            # Save updated user data
            users[username] = user
            self.save_users(users)
            
            return user, None
        else:
            # Increment failed attempts
            user.failed_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_attempts >= 5:
                user.locked_until = (datetime.utcnow() + timedelta(minutes=15)).isoformat()
                message = "Too many failed attempts. Account locked for 15 minutes."
            else:
                remaining = 5 - user.failed_attempts
                message = f"Invalid username or password. {remaining} attempts remaining."
            
            # Save updated user data
            users[username] = user
            self.save_users(users)
            
            return None, message
    
    def create_user(self, username, password, role='user'):
        """Create a new user (superadmin only)"""
        if not username or not password:
            return False, "Username and password are required"
        
        users = self.load_users()
        
        if username in users:
            return False, "User already exists"
        
        # Validate password strength
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        # Create new user
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password, method='pbkdf2:sha256', salt_length=16),
            role=role
        )
        
        users[username] = new_user
        self.save_users(users)
        
        return True, "User created successfully"
    
    def update_user(self, username, password=None, role=None, is_active=None):
        """Update user details (superadmin only)"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        user = users[username]
        
        if password:
            if len(password) < 8:
                return False, "Password must be at least 8 characters long"
            user.password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        
        if role:
            user.role = role
        
        if is_active is not None:
            user._is_active = is_active
        
        users[username] = user
        self.save_users(users)
        
        return True, "User updated successfully"
    
    def delete_user(self, username):
        """Delete a user (superadmin only)"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        # Prevent deleting the last superadmin
        user = users[username]
        if user.role == 'superadmin':
            superadmin_count = sum(1 for u in users.values() if u.role == 'superadmin')
            if superadmin_count <= 1:
                return False, "Cannot delete the last superadmin"
        
        del users[username]
        self.save_users(users)
        
        return True, "User deleted successfully"
    
    def get_all_users(self):
        """Get all users (for superadmin view)"""
        users = self.load_users()
        return list(users.values())

# Decorator for superadmin-only routes
def superadmin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_superadmin():
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Session security helper
def setup_session_security(app):
    """Additional session security measures"""
    @app.before_request
    def check_session_security():
        # Skip security checks for login page
        if request.endpoint == 'login':
            return
        
        # Session activity tracking removed - no forced re-authentication
        
        # Check for session hijacking attempts (user agent change)
        if 'user_agent' in session:
            if session['user_agent'] != request.headers.get('User-Agent'):
                session.clear()
                return redirect(url_for('login'))
        else:
            session['user_agent'] = request.headers.get('User-Agent') 