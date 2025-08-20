# Authentication System Setup Guide

## Overview

This platform now includes a comprehensive security-focused authentication system with the following features:

- **Secure Login System**: Password-protected access with encrypted credentials
- **Role-Based Access Control**: Superadmin and regular user roles
- **User Management**: Superadmins can create, edit, and delete user accounts
- **Session Security**: Automatic timeout, session hijacking protection
- **Account Lockout**: Protection against brute-force attacks (5 attempts = 15 min lockout)
- **Encrypted User Storage**: All user data is encrypted at rest

## Initial Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the template file and edit it:

```bash
cp env.template .env
```

Edit `.env` with your secure credentials:

```env
# REQUIRED: Change these immediately!
SUPERADMIN_USERNAME=admin
SUPERADMIN_PASSWORD=YourSecurePassword123!

# Optional: Add additional users
USER1_USERNAME=teacher1
USER1_PASSWORD=AnotherSecurePassword123!
```

### 3. First Login

1. Start the application: `python app.py`
2. Navigate to the login page (any URL will redirect you there)
3. Login with your superadmin credentials
4. **IMPORTANT**: Change the default password immediately

## Security Features

### Password Requirements
- Minimum 8 characters
- Recommended: Mix of uppercase, lowercase, numbers, and symbols

### Session Security
- Sessions expire after 8 hours by default
- "Remember me" extends session to 24 hours
- 30 minutes of inactivity triggers re-authentication
- Session hijacking protection (user agent verification)

### Account Protection
- 5 failed login attempts = 15-minute account lockout
- Superadmins can manually unlock accounts
- Encrypted password storage using PBKDF2-SHA256

### Data Encryption
- User database is encrypted using Fernet (symmetric encryption)
- Encryption key is generated on first run and stored securely

## User Management (Superadmin Only)

### Access User Management
1. Login as superadmin
2. Navigate to Settings > User Management
3. Or use the dropdown menu in the navbar

### Create New Users
1. Click "Create New User"
2. Enter username (alphanumeric, underscore, hyphen only)
3. Set a secure password (minimum 8 characters)
4. Choose role (User or Superadmin)

### Manage Existing Users
- **Edit**: Change user role or active status
- **Reset Password**: Set a new password for any user
- **Unlock**: Remove lockout after failed login attempts
- **Delete**: Remove user (cannot delete last superadmin)

## Environment Variables

### Required
- `SUPERADMIN_USERNAME`: Default admin username
- `SUPERADMIN_PASSWORD`: Default admin password

### Optional
- `USER1_USERNAME` to `USER9_USERNAME`: Pre-configured usernames
- `USER1_PASSWORD` to `USER9_PASSWORD`: Pre-configured passwords
- `SECRET_KEY`: Flask session secret (auto-generated if not set)

### Security Settings (Advanced)
- `SESSION_COOKIE_SECURE`: Set to True for HTTPS only (default: True)
- `SESSION_COOKIE_HTTPONLY`: Prevent JavaScript access (default: True)
- `SESSION_COOKIE_SAMESITE`: CSRF protection (default: Lax)
- `PERMANENT_SESSION_LIFETIME`: Default session timeout in seconds (default: 28800, extended to 86400 with "Remember me")

## File Structure

```
.auth_key          # Encryption key (auto-generated, DO NOT SHARE)
users.json         # Encrypted user database (DO NOT EDIT MANUALLY)
auth.py            # Authentication module
templates/
  login.html       # Login page
  users.html       # User management interface
```

## Security Best Practices

1. **Change Default Credentials**: Always change the default superadmin password
2. **Use Strong Passwords**: Minimum 12 characters with mixed case and symbols
3. **Regular Updates**: Keep dependencies updated for security patches
4. **Backup Encryption Key**: Save `.auth_key` securely (losing it = losing access)
5. **HTTPS in Production**: Always use HTTPS in production environments
6. **Environment Variables**: Never commit `.env` to version control

## Troubleshooting

### Forgot Password
- Ask a superadmin to reset your password
- If all superadmin access is lost, you'll need to reset the system

### Account Locked
- Wait 15 minutes for automatic unlock
- Ask a superadmin to unlock immediately

### Lost Encryption Key
- If `.auth_key` is lost, user database cannot be decrypted
- You'll need to delete `users.json` and restart (loses all users)

## Support

For security concerns or questions, please contact your system administrator. 