# CS50Trader
#### Video Demo:  [URL HERE](https://youtu.be/5JyukljwI5o)
CS50Trader is a real-time cryptocurrency trading bot and management system. It integrates machine learning, real-time data processing, and a user-friendly interface to provide an all-in-one solution for cryptocurrency traders. It is implemented with a modular architecture and focuses on high performance, scalability, and user experience.

---

## Table of Contents
- [Introduction](#introduction)
  - [Features](#features)
- [System Architecture](#system-architecture)
  - [Backend](#backend)
  - [BotCore](#botcore)
  - [Frontend](#frontend)
- [Distinctiveness and Complexity](#distinctiveness-and-complexity)
- [Installation and Setup](#installation-and-setup)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Setup Backend](#3-setup-backend)
  - [4. Run BotCore](#4-run-botcore)
  - [5. Start Frontend](#5-start-frontend)
- [Database Design](#database-design)
- [Machine Learning and Prediction](#machine-learning-and-prediction)
- [Third-Party Libraries and Tools](#third-party-libraries-and-tools)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction
CS50Trader is a cryptocurrency trading bot built as part of a final capstone project for CS50. It features three main modules:
1. **Backend**: Manages user authentication, database interaction, and API endpoints.
2. **BotCore**: Contains trading logic, market data fetching, and prediction algorithms.
3. **Frontend**: A responsive user interface built with HTML, TailwindCSS, and JavaScript.

### Features
- **User Authentication**: Secure signup, login, and token-based authentication.
- **Trading Bot**:
  - Predicts market patterns using machine learning (Voting Classifier).
  - Executes trades in real time using the CCXT Python library and WebSocket API.
  - Multi-threaded architecture for handling multiple coins simultaneously.
- **Dashboard**:
  - Displays performance metrics and trade charts for the last 7 and 14 days.
  - Lists open trades and their statuses.
- **Customizable Settings**: Show timeframe, leverage, and stake amount.
- **Trade History**:
  - View paginated trade history with a limit of 5 trades per page.
  - Filter trades by single date and pair.
- **Account Management**: Change password and add account balance.

---

## System Architecture

### Backend
- Built with **Python Flask** and follows a **Modular Blueprint Architecture**.
- **Database**: Uses SQLite3 with two tables:
  - `users`: Stores user account details.
  - `trades`: Tracks trade entries, exits, and profits.
- Key endpoints:
  - **Authentication**: `/auth/signup`, `/auth/login`, `/auth/checktoken`
  - **Dashboard**: `/dashboard/chart`, `/dashboard/open_trades`
  - **History**: `/history/pair_list`, `/history/trade_history`
  - **Account Management**: `/account/change_password`, `/account/add_balance`
- Backend logic includes a custom Object-Relational Mapping (ORM) and a `calculate_profit` function (excluding exchange fees).

### BotCore
- **Trade Execution**:
  - Fetches market data using CCXT Python library.
  - Predicts zig-zag patterns using a **Voting Classifier** (KNN and CatBoost models).
  - Confirms trends using **AdaptiveTrendFinder**, optimized with Numba.
- **Features**:
  - Custom feature calculation inspired by PineScript (converted to Python with NumPy and Pandas).
  - Supports fetching over 1,000 historical candles by concatenating requests.
  - Multi-threaded for handling multiple trading pairs in real time.
- **Machine Learning**:
  - Training data:
    - Features: 5 custom-calculated features.
    - Labels: ZigZag++ (adapted from PineScript).
  - Libraries: Scikit-learn, CatBoost, NumPy, Pandas.

### Frontend
- Built with **HTML**, **Tailwind CSS**, and **JavaScript**.
- Features:
  - **Login Page**: Redirects unauthenticated users to the login page. Includes signup functionality.
  - **Dashboard**: Displays charts and performance metrics. Fetches data dynamically from the backend.
  - **Trade Settings**: Allows users to configure timeframe, leverage, and stake amount.
  - **History Page**: Paginated view of trade history with filters.
  - **Account Page**: Enables password changes and balance additions.
- All actions are performed dynamically using JSON data, ensuring no page reloads after the initial load.

---

## Distinctiveness and Complexity
This project stands out for its advanced architecture and real-world application in cryptocurrency trading:
1. **Distinctiveness**:
   - Combines trading bot logic with a user-facing web interface.
   - Integrates machine learning for predictive trading.
2. **Complexity**:
   - Implements custom PineScript-inspired feature calculations.
   - Uses multi-threading, real-time WebSocket data, and complex ML models.
   - Custom ORM and modular Flask architecture.

---

## Installation and Setup

### 1. Clone Repository
```bash
git clone https://github.com/pyaeheintun/CS50Trader.git
cd CS50Trader
```

### 2. Install Dependencies
Set up a virtual environment and install Python dependencies:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Run Backend
Run Flask Backend:
```bash
python backend/app.py
```

### 4. Run BotCore
Run Botcore Trade Bot:
```bash
python botcore/main.py
```

### 5. Start Frontend
Simply open `index.html` in a web browser to access the frontend interface.

---

## Database Design
- **users**:
  - `id`: Primary key
  - `username`: Unique
  - `password`: Hashed password
  - `balance`: User balance
- **trades**:
  - `id`: Primary key
  - `user_id`: Foreign key to `users`
  - `pair`: Trading pair (e.g., BTC/USD)
  - `entry_price`: Price at entry
  - `exit_price`: Price at exit
  - `is_short`: Long or Short boolean
  - `created_at` : Unix timestamp
  - `updated_at`: Unix timestamp
  - `leverage`: Borrowed money from exchange
  - `stake_ammount` : Ammount for each Trade
  - `quantity` : Quantify of a coin for each Trade
  - `is_completed` : to identify Open Trades and Closed Trades

---

## Machine Learning and Prediction
1. **Custom Features**:
   - Inspired by PineScript logic.
   - Calculated using NumPy and Pandas for performance.
2. **Models**:
   - **VotingClassifier**: Combines KNN and CatBoost for zig-zag pattern prediction.
3. **Trend Confirmation**:
   - **AdaptiveTrendFinder**: Optimized using Numba for real-time processing.

---

## Third-Party Libraries and Tools
- **Backend**:
  - Flask, SQLite3
  - Flask-RESTful for APIs
- **BotCore**:
  - CCXT (WebSocket data fetching)
  - Scikit-learn, CatBoost, Numba
- **Frontend**:
  - Tailwind CSS, Chart.js

---

## License
This project is licensed under the MIT License.

---

## Acknowledgements
- PineScript inspirations:
  - Lorentization_classification by jdehorty.
  - ZigZag++ by Devlucem.
  - AdaptiveTrendFinder by Julen_Eche.
- CS50 and its instructors for the foundation.