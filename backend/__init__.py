# app/__init__.py
from flask import Flask, jsonify
from flask_cors import CORS

ALLOWED_ORIGINS = [
    'https://wave-crafter.com',
    'https://wave-crafter-587074aad3d2.herokuapp.com',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:5000',
]


def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
    app.config['MAX_FORM_MEMORY_SIZE'] = 500 * 1024 * 1024  # 500 MB for form fields
    app.config['MAX_FORM_PARTS'] = 1000

    # CORS configuration
    CORS(app,
         origins=ALLOWED_ORIGINS,
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type'],
         expose_headers=['Content-Type'],
         max_age=3600)

    # Error handlers with CORS headers (use same restricted origins, not wildcard)
    def _add_cors_headers(response):
        from flask import request as req
        origin = req.headers.get('Origin', '')
        if origin in ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.errorhandler(503)
    def service_unavailable(e):
        response = jsonify({
            'error': 'Service temporarily unavailable. The server is waking up, please try again in a moment.'
        })
        response.status_code = 503
        return _add_cors_headers(response)

    @app.errorhandler(500)
    def internal_error(e):
        response = jsonify({
            'error': 'Internal server error'
        })
        response.status_code = 500
        return _add_cors_headers(response)

    from .routes import main
    app.register_blueprint(main)

    return app