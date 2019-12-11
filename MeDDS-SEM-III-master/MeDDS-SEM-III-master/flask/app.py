import flask
from flask import *

from flask_cors import CORS


from get_model import mask

app = Flask(__name__)
CORS(app)

@app.route('/query')
def path():
    path = request.args.get('path')
    filepath,status = mask('./in/',path+'.png')

    if request.method == 'GET':
        if status == True:
            return path
    abort(404)