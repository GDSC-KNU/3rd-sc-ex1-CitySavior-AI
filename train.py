import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

def get_image_labels(root_dir):
    # 이미지 ID와 라벨 매핑을 위한 딕셔너리 생성
    image_labels = {'file_name': [],'image_id': [], 'class': [], 'degree': []}
    attrib_list = ['image_id', 'class', 'status', 'damagetype', 'degree']
    attirb_dict = {'image_id':None, 'class': None, 'status': None, 'damagetype': None, 'degree': None}
    for folder in os.listdir(root_dir):
        folder_dir = os.path.join(root_dir, folder)
        for label in os.listdir(folder_dir):
            label_dir = os.path.join(folder_dir, label)
            if '[라벨]' in label:
                for file in os.listdir(label_dir):
                    # JSON 파일 로드
                    file_dir = os.path.join(label_dir, file)
                    with open(file_dir) as f:
                        data = json.load(f)
                    # 이미지 ID와 라벨 매핑
                    for annotation in data['annotations']:
                        attributes = annotation['attributes']
                        # 'class', 'status', 'damagetype', 'degree' 키 확인 및 값 가져오기
                        for attrib in attrib_list:
                            if attrib not in attributes:
                                continue
                            else:
                                attirb_dict[attrib] = attributes.get(attrib)   
                        
                    # 클래스와 상태를 결합하여 라벨 생성 (None인 경우는 공백으로 처리)
                    image_labels['image_id'].append(annotation['image_id']) 
                    image_labels['class'].append(attirb_dict['class'])
                    image_labels['degree'].append(attirb_dict['degree'])
                    # 이미지 파일 이름과 라벨 매핑
                    image_file_labels = {}
                    for image in data['images']:
                        image_id = image['id']
                        file_name = image['file_name']

                        # 이미지 ID를 사용하여 라벨 찾기
                        label = image_labels.get('image_id')[-1]
                        if label == image_id:
                            image_labels['file_name'].append(file_name)
            else:
                continue
    del image_labels['image_id']    
    return image_labels

image_labels = get_image_labels('Dataset\Training')
# 데이터프레임 생성
df = pd.DataFrame(image_labels)
df.to_csv('image_labels.csv', index=False)
print(df)
# # 클래스 라벨을 숫자로 변환
# le = LabelEncoder()
# df['class'] = le.fit_transform(df['class'])

# # ImageDataGenerator 생성
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# # Train 데이터 불러오기
# train_generator = datagen.flow_from_dataframe(
#     df, 
#     directory='Training/', 
#     x_col='filename',
#     y_col='class',
#     target_size=(150, 150), 
#     batch_size=32,
#     class_mode='categorical',
#     subset='training')

# # Validation 데이터 불러오기
# validation_generator = datagen.flow_from_dataframe(
#     df,
#     directory='Validation/', 
#     x_col='filename',
#     y_col='class',
#     target_size=(150, 150), 
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation')

# # CNN 모델 구성
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(len(df['class'].unique()), activation='softmax')
# ])

# # 모델 컴파일
# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
#               metrics=['accuracy'])

# # 모델 학습
# history = model.fit(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=30,
#       validation_data=validation_generator,
#       validation_steps=50,
#       verbose=2)
