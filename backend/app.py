import asyncio
from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import timedelta
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flask import g

from blueprint.auth.app import app as auth_app
from blueprint.dashboard.app import app as dashboard_app
from blueprint.history.app import app as history_app
from blueprint.account.app import app as account_app

from blueprint.auth.helper import BcryptHelper
from model.db_helper import create_necessary_table
from flask_cors import CORS

# App Starting Point
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)
# Secret key and JWT configuration
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a secure key
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret'  # Change this to a secure key
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

jwt = JWTManager(app)
BcryptHelper.create_bcrypt(app)

# Register Blueprints
app.register_blueprint(auth_app,url_prefix="/auth")
app.register_blueprint(dashboard_app,url_prefix="/dashboard")
app.register_blueprint(history_app,url_prefix="/history")
app.register_blueprint(account_app,url_prefix="/account")

with app.app_context():
    create_necessary_table()

# Close Database Connection when app is down
@app.teardown_appcontext
def close_connection(exception) -> None:
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

if __name__ == '__main__':
    app.run(debug=True)