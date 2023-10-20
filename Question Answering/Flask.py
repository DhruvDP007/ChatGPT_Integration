from flask import Flask, request, render_template, jsonify
# import subprocess
from Helper import get_answer
import json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/run', methods=['GET', 'POST'])
def helper_run():
    if request.method == 'POST':
        data = json.loads(request.data)
        # print(data)
        result = get_answer(data['question'])
        print(result)
        return jsonify(result)
    else:
        return jsonify({})
    

if __name__ == '__main__':
        app.run(debug=True,host="0.0.0.0",use_reloader=False)



