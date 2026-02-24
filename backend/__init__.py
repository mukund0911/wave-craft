# app/__init__.py
from flask import Flask, jsonify
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
    app.config['MAX_FORM_MEMORY_SIZE'] = 500 * 1024 * 1024  # 500 MB for form fields
    app.config['MAX_FORM_PARTS'] = 1000

    # CORS configuration - allow all origins to handle dyno wake-up issues
    CORS(app,
         origins=['https://wave-crafter.com',
                  'https://wave-crafter-3ad7c939239a.herokuapp.com',
                  'http://localhost:3000',
                  'http://127.0.0.1:3000',
                  'http://localhost:5000'],
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type'],
         supports_credentials=True,
         expose_headers=['Content-Type'])

    # Error handlers with CORS headers
    @app.errorhandler(503)
    def service_unavailable(e):
        response = jsonify({
            'error': 'Service temporarily unavailable. The server is waking up, please try again in a moment.'
        })
        response.status_code = 503
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.errorhandler(500)
    def internal_error(e):
        response = jsonify({
            'error': 'Internal server error'
        })
        response.status_code = 500
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    from .routes import main
    app.register_blueprint(main)

    return app