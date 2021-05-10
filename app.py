from flask import Flask,request,jsonify
from flask_cors import CORS
import tmswdr
app = Flask(__name__)
CORS(app) 
        
@app.route('/recommendationsuit', methods=['GET'])
def recommendsuit():
    result=tmswdr.recommend_suit(request.args.get('Design_codein'))
    return jsonify(result)

@app.route('/recommendationhalfsuit', methods=['GET'])
def recommendhalfsuit():
    result=tmswdr.recommend_halfsuit(request.args.get('Design_codein'))
    return jsonify(result)

@app.route('/recommendation3piecesuit', methods=['GET'])
def recommend3piecesuit():
    result=tmswdr.recommend_3piecesuit(request.args.get('Design_codein'))
    return jsonify(result)


@app.route('/samplesuit')
def getsamplesuit():
    result=tmswdr.sample_suit()
    return jsonify(result)

@app.route('/samplehalfsuit')
def getsamplehalfsuit():
    result=tmswdr.sample_halfsuit()
    return jsonify(result)

@app.route('/sample3piecesuit')
def getsample3piecesuit():
    result=tmswdr.sample_3piecesuit()
    return jsonify(result)

@app.route('/')
def test():
    return "Hurray!, Recommendation System is Working..."
if __name__=='__main__':
    app.run(port = 5000, debug =True)

