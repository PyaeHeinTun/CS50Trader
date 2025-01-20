from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,decode_token
)
from .helper import BcryptHelper
from model.user_model import User

app = Blueprint('auth_app', __name__)

# Route: Signup
@app.route('/signup', methods=['POST'])
def signup() -> dict:
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get('password')
        confirm = request.form.get("confirm")

        if not username or not password or (confirm != password):
            return jsonify({"title":"Could not SignUp.","message": "username or password is missing."}), 400 
        user = User.get_user_by_username(username)
        if user:
            return jsonify({"title":"Could not SignUp.","message": "User already exists!"}), 400

        hashed_password = BcryptHelper.create_hash_password(password)
        user = User(username,hashed_password)
        user.save()
        access_token = create_access_token(identity=username)
        response = jsonify({"signup": True, "access_token": access_token})
        return response, 200
    return jsonify({"message": "Method not allowed."}), 404


# Route: Login
@app.route('/login', methods=['POST'])
def login() -> dict:
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return jsonify({"title":"Could not SignIn.","message": "username or password is missing."}), 400 
        user = User.get_user_by_username(username)
        # Validate user credentials
        if not user or not BcryptHelper.check_password(user.password, password):
            return jsonify({"title":"Could not SignIn.","message": "Invalid username or password!"}), 400

        # Generate JWT token
        access_token = create_access_token(identity=username)
        response = jsonify({"login": True, "access_token": access_token,"username":username})
        return response,200
    
    return jsonify({"message":"Method not allowed.","status":404}),404


# Token Validity Check
@app.route("/checktoken",methods=["POST"])
def checktoken() -> dict:
    if request.method == "POST":
        token = request.form.get("token")
        if not token:
            return jsonify({"title":"Token is Empty","message": "Must provide token in form to check."}), 400 

        decoded_token = decode_token(token)
        return jsonify({"isValid":True,"username":decoded_token['sub']}), 200
    return jsonify({"message":"Method not allowed.","token":"","status":200}),404

# Logout Route
@app.route('/logout')
def logout() -> dict:
    response = jsonify({"logout": True})
    return response
