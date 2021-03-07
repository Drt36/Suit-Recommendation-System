from flask import Flask,request,jsonify
from flask_cors import CORS
import tmswdr
app = Flask(__name__)
CORS(app) 
        
@app.route('/recommendation', methods=['GET'])
def recommend():
    result=tmswdr.recommend_cosine(request.args.get('Design_codein'))
    return jsonify(result)

@app.route('/')
def test():
    return "Hurray!, Working..."
if __name__=='__main__':
    app.run(port = 5000, debug =True)

