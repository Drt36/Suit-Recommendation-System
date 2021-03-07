from flask import Flask,request,jsonify
from flask_cors import CORS
import tmswdr
app = Flask(__name__)
CORS(app) 
        
@app.route('/recommendation', methods=['GET'])
def recommend():
    result=tmswdr.recommend_cosine(request.args.get('Design_codein'))
    return jsonify(result)

@app.route('/sample', methods=['GET'])
def getsample():
    result=tmswdr.sample_row()
    return jsonify(result)

@app.route('/')
def test():
    return "Hurray!, Recommendation System is Working..."
if __name__=='__main__':
    app.run(port = 5000, debug =True)

