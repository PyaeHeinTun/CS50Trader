from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,decode_token
)
from model.user_model import User
from blueprint.auth.helper import BcryptHelper

app = Blueprint('account_app', __name__)

@app.route("change_password",methods=["POST"])
def change_password() -> dict:
    if request.method == "POST":
        token = request.form.get("token")
        password = request.form.get("password")
        if not token or not password:
            return jsonify({"title":"Could not Change Password.","message": "token or password is missing."}), 400 
        
        decoded_token = decode_token(token)
        username = decoded_token['sub']
        user = User.get_user_by_username(username)
        if not user:
            return jsonify({"title":"User could not find in database.","message": "Must provide valid token in form."}), 400 
        
        hashed_password = BcryptHelper.create_hash_password(password)
        user.password = hashed_password
        user.update()
        return jsonify({"title":"Success","message":"Successfully updated."})
    return jsonify({"message": "Method not allowed."}), 404

@app.route("add_balance",methods=["POST"])
def addBalance() -> dict:
    if request.method == "POST":
        token = request.form.get("token")
        balance = request.form.get("balance")
        if not token or not balance:
            return jsonify({"title":"Could not Add Balance.","message": "token or balance is missing."}), 400 
        
        balance = int(balance)
        decoded_token = decode_token(token)
        username = decoded_token['sub']
        user = User.get_user_by_username(username)
        if not user:
            return jsonify({"title":"User could not find in database.","message": "Must provide valid token in form."}), 400 
        
        user.balance = user.balance+balance
        user.update()
        return jsonify({"message":"Successfully updated."})
    return jsonify({"message": "Method not allowed."}), 404 