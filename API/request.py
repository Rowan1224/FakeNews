import pickle
import re
import string
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

punctuation_list = ["।", "”", "“", "’"]
for p in string.punctuation.lstrip():
    punctuation_list.append(p)


def prediction(txt):
    infile = open("tfidf_char_pkl", 'rb')
    tfidf_char = pickle.load(infile)
    infile.close()
    x = tfidf_char.transform([txt])
    # print(x.shape)
    infile = open("model", 'rb')
    clf = pickle.load(infile)
    infile.close()
    y_pred = clf.predict(x)
    print(y_pred)
    return y_pred[0]


def clean(doc):
    for p in punctuation_list:
        doc = doc.replace(p, "")
    doc = re.sub(r'[\u09E6-\u09EF]', "", doc, re.DEBUG)  # replace digits
    # doc = doc.replace("\n", "")

    return doc


# with open("input.txt", 'r') as infile:
#     doc = ""
#     for line in infile:
#         doc = doc + line.replace("\n", "")


# txt = clean(doc)
# prediction(txt)

app = Flask(__name__)
CORS(app)


@app.route("/check", methods=['POST'])
def predict():
    doc = request.json['data']
    txt = clean(doc)

    infile = open("tfidf_char_pkl", 'rb')
    tfidf_char = pickle.load(infile)
    infile.close()
    x = tfidf_char.transform([txt])
    # print(x.shape)
    infile = open("model", 'rb')
    clf = pickle.load(infile)
    infile.close()
    y_pred = clf.predict(x)
    print(y_pred)
    ret = '{"prediction":' + str(float(y_pred)) + '}'

    return ret


# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
