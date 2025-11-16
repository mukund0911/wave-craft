# app/__init__.py
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app,
         origins=['https://wave-crafter.com', 'http://localhost:3000', 'http://127.0.0.1:3000'],
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type'],
         supports_credentials=True,
         expose_headers=['Content-Type'])

    from .routes import main
    app.register_blueprint(main)

    return app
