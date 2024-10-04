from backend import create_app
from backend.routes import socketio

app = create_app()
socketio.init_app(app)

if __name__ == '__main__':
    socketio.run(app)
