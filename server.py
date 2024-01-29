from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# 모델 로드
model = load_model('model/ResNet50V2_fine_tuned.h5')

# 이미지 전처리 함수
def preprocess_image(image):
    # 이미지를 모델 입력 크기에 맞게 조정
    image = image.resize((224, 224))
    # 이미지를 numpy 배열로 변환
    image_array = np.array(image)
    # 모델 입력 형식에 맞게 차원 추가
    image_array = np.expand_dims(image_array, axis=0)
    # 이미지 정규화
    image_array = image_array / 255.0
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    # POST 요청에서 이미지 파일 받기
    file = request.files['image']
    # 이미지 파일을 PIL 이미지로 열기
    image = Image.open(file)
    # 이미지 전처리
    processed_image = preprocess_image(image)
    # 예측
    predictions = model.predict(processed_image)
    # 예측 결과를 파이썬 기본 데이터 타입으로 변환
    predictions = predictions.tolist()
    # 예측 결과 반환
    print(predictions)
    argmax = np.argmax(predictions[0])
    result = str(argmax)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
