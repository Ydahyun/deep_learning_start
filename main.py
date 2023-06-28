import numpy as np
import pandas as pd
import tensorflow as tf


# 데이터 전처리
data = pd.read_csv('gpascore.csv')  # .csv 파일 읽겠다.
# print(data.isnull().sum())  # 데이터 빵꾸 찾아줌
data = data.dropna()  # NaN/빈 값있는 행을 제거해줌.
# data = data.fillna(100)  # 빈 칸을 100으로 채워줌.

y_data = data['admit'].values
x_data = []

# x_data 에 리스트 형식으로 값을 넣어줌.
for i, rows in data.iterrows():
    x_data.append([ rows['gre'], rows['gpa'], rows['rank'] ])


# 딥러닝 모델 디자인
model = tf.keras.models.Sequential([
    # 레이어 갯수/노드갯수는 마음대로 기준없음 결과 잘나올 때까지 실험으로 파악해야함.
    # 관습적으로 2의 제곱수로 함.
    tf.keras.layers.Dense(64, activation='tanh'),   # 이 한 줄이 레이어 하나
    tf.keras.layers.Dense(128, activation='tanh'),  # 덴스안의 값은 노드의 개수
    tf.keras.layers.Dense(1, activation='sigmoid'),    # 마지막 레이어는 노드 1개.
])
# 모델 컴파일 하기
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 손실함수 loss='binary_crossentropy' 는 결과가 0과 1사이의 분류/확률문제에서 씀


# 모델 학습(fit) 시키기
model.fit(np.array(x_data), np.array(y_data), epochs=3000)


# 예측
res = model.predict([ [750, 3.70, 3], [400, 2.2, 1] ])
print(res)