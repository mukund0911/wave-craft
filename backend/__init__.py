# app/__init__.py
from flask import Flask, jsonify, request
from flask_cors import CORS

ALLOWED_ORIGINS = [
    'https://wave-crafter.com',
    'https://wave-crafter-587074aad3d2.herokuapp.com',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:5000',
]


def _get_cors_origin():
    """Return the request origin if it's in the allowed list, else the first allowed origin."""
    origin = request.headers.get('Origin', '')
    if origin in ALLOWED_ORIGINS:
        return origin
    return ALLOWED_ORIGINS[0]


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
         supports_credentials=True,
         expose_headers=['Content-Type'])

    # Error handlers with CORS headers (must match origin, not wildcard, when credentials are used)
    @app.errorhandler(503)
    def service_unavailable(e):
        response = jsonify({
            'error': 'Service temporarily unavailable. The server is waking up, please try again in a moment.'
        })
        response.status_code = 503
        response.headers['Access-Control-Allow-Origin'] = _get_cors_origin()
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.errorhandler(500)
    def internal_error(e):
        response = jsonify({
            'error': 'Internal server error'
        })
        response.status_code = 500
        response.headers['Access-Control-Allow-Origin'] = _get_cors_origin()
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    from .routes import main
    app.register_blueprint(main)

    return app