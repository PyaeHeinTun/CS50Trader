from typing import Self
from .db_helper import get_db
import time
import pytz

class User():
    def __init__(self,username,password,balance=0) -> None:
        self.username = username
        self.password = password
        self.balance = balance
        self.created_at = int(time.time())

    # @staticmethod
    def get_users_by_id(user_id) -> Self:
        if user_id:
            user = get_db().cursor().execute("SELECT * FROM users WHERE id=?",(str(user_id))).fetchall()
            return User(*user[0][1:-1])
        
    def get_user_by_username(username) -> Self:
        if username:
            user = get_db().cursor().execute("SELECT * FROM users WHERE username=?",(username,)).fetchall()
            if len(user) > 0:
                return User(*user[0][1:-1])
        return None

    def save(self) -> None:
        if self.username and self.password:
            get_db().cursor().execute("INSERT INTO users (username,password,balance,created_at) VALUES(?,?,?,?)",(self.username,self.password,self.balance,self.created_at))
            get_db().cursor().connection.commit()

    def update(self) -> None:
        self.created_at = int(time.time())
        get_db().cursor().execute("UPDATE users SET username=?,password=?,balance=?,created_at=?",(self.username,self.password,self.balance,self.created_at))
        get_db().cursor().connection.commit()

    def __repr__(self:Self):
        return f"{self.username} {self.balance}"
