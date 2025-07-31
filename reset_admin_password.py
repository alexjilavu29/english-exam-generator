import os
import json
import argparse
from cryptography.fernet import Fernet, InvalidToken
from werkzeug.security import generate_password_hash

USERS_FILE = 'users.json'
KEY_FILE = '.auth_key'

def load_key():
    """Loads the encryption key from .auth_key."""
    if not os.path.exists(KEY_FILE):
        raise FileNotFoundError(f"Encryption key file not found at '{KEY_FILE}'. Make sure the app has run and generated it.")
    with open(KEY_FILE, 'rb') as f:
        return f.read()

def load_users(cipher_suite):
    """Loads and decrypts users from users.json."""
    if not os.path.exists(USERS_FILE):
        raise FileNotFoundError(f"Users file not found at '{USERS_FILE}'.")
    with open(USERS_FILE, 'rb') as f:
        encrypted_data = f.read()
    
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        return json.loads(decrypted_data)
    except InvalidToken:
        raise ValueError("Failed to decrypt users.json. The encryption key in .auth_key may be incorrect or corrupted.")
    except json.JSONDecodeError:
        raise ValueError("Failed to parse user data after decryption. The users.json file might be corrupted.")

def save_users(users_data, cipher_suite):
    """Encrypts and saves users back to users.json."""
    serialized_data = json.dumps(users_data, indent=4).encode('utf-8')
    encrypted_data = cipher_suite.encrypt(serialized_data)
    with open(USERS_FILE, 'wb') as f:
        f.write(encrypted_data)

def main():
    """Main function to reset the password."""
    parser = argparse.ArgumentParser(description="Reset the password for a superadmin account.")
    parser.add_argument('new_password', type=str, help="The new password for the admin account.")
    parser.add_argument('--username', type=str, default='admin', help="The username of the superadmin account (default: 'admin').")
    args = parser.parse_args()

    new_password = args.new_password
    admin_username = args.username

    if len(new_password) < 8:
        print("Error: Password must be at least 8 characters long.")
        return

    try:
        print("Attempting to reset password...")
        key = load_key()
        cipher_suite = Fernet(key)
        users = load_users(cipher_suite)

        if admin_username not in users:
            print(f"Error: Superadmin user '{admin_username}' not found in {USERS_FILE}.")
            return
        
        print(f"Found user '{admin_username}'. Updating password and unlocking account...")
        
        new_hash = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=16)
        users[admin_username]['password_hash'] = new_hash
        users[admin_username]['failed_attempts'] = 0
        users[admin_username]['locked_until'] = None
        
        save_users(users, cipher_suite)
        
        print(f"\nSuccess! The password for '{admin_username}' has been reset.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you are running this script from the app's root directory on PythonAnywhere,")
        print("and that 'users.json' and '.auth_key' exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    main() 