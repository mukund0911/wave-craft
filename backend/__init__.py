# app/__init__.py
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app, resources={
            r"/*": {
                "origins": [
                    "http://wave-crafter.com",
                    "http://localhost:5000"  # for local development
                ]
            }
        })

    from .routes import main
    app.register_blueprint(main)

    return app
