from flask_bcrypt import Bcrypt

from flask_bcrypt import Bcrypt


class BcryptHelper:
    bcrypt = None 

    @staticmethod
    def create_bcrypt(app) -> None:
        """
        Initializes the Bcrypt instance with the Flask app.
        """
        BcryptHelper.bcrypt = Bcrypt(app)

    @staticmethod
    def create_hash_password(password: str) -> str:
        """
        Hashes a plain-text password.
        """
        if BcryptHelper.bcrypt is None:
            raise Exception("Bcrypt is not initialized. Call 'create_bcrypt(app)' first.")
        return BcryptHelper.bcrypt.generate_password_hash(password).decode('utf-8')

    @staticmethod
    def check_password(hashed_password: str, password: str) -> bool:
        """
        Verifies if a plain-text password matches the hashed password.
        """
        if BcryptHelper.bcrypt is None:
            raise Exception("Bcrypt is not initialized. Call 'create_bcrypt(app)' first.")
        return BcryptHelper.bcrypt.check_password_hash(hashed_password, password)
