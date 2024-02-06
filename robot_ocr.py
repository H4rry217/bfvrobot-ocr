import io
import json
import easyocr
from flask import Flask, request, jsonify
import numpy
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity


import jieba
import jieba.analyse

# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# pip uninstall torch torchvision torchaudio
# pyinstaller --onefile robot_ocr.py

OCR_READER = easyocr.Reader(lang_list=['en', 'ch_sim'], model_storage_directory="./")
W2V_MODEL = KeyedVectors.load_word2vec_format("45000-small.txt")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
app = Flask(__name__)   

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        image_file = request.files['image']
        
        input_bytes = image_file.read()
        byte_data = io.BytesIO(input_bytes).getvalue()
        results = OCR_READER.readtext(byte_data)

        return jsonify({"result": json.loads(json.dumps(results, cls=NumpyEncoder, ensure_ascii=False))})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/similar', methods=['POST'])
def similar():
    try:
        data = request.json
        
        findword = data["word"]
        words = data["search"]

        findword_extract = keyword_extract(findword);

        closest_word = None
        max_similarity = 0

        for word in words:
            extract_words = keyword_extract(word)
            similarity = get_similarity(findword_extract, extract_words)
            if similarity > max_similarity:
                max_similarity = similarity
                closest_word = word

        return jsonify({"result": closest_word, "similarity": float(max_similarity)})
    except Exception as e:
        return jsonify({'error': str(e)})
    
def keyword_extract(content):
    return jieba.analyse.extract_tags(content, topK=10)

def get_similarity(keywords_1, keywords_2):
    empty_vector = numpy.zeros(W2V_MODEL.vector_size)

    if len(keywords_1) == 0 or len(keywords_2) == 0:
        return 0.0
    
    kw1_vector = sum(W2V_MODEL[keyword] if keyword in W2V_MODEL else empty_vector for keyword in keywords_1) / len(keywords_1)
    kw2_vector = sum(W2V_MODEL[keyword] if keyword in W2V_MODEL else empty_vector for keyword in keywords_2) / len(keywords_2)

    similarity = cosine_similarity([kw1_vector], [kw2_vector])[0][0]
    return similarity

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=11452)