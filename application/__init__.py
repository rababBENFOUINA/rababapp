from flask import Flask

app= Flask(__name__)

from application.routes import main
from application.routes import formulaire

