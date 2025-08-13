import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import rc

# 한글 폰트
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터
lengths = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
           31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
           35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
           10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
weights = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
           500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
           700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
           7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

X = np.column_stack((lengths, weights))
y = np.concatenate((np.ones(35), np.zeros(14)))

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# KNN 모델
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 수상한 도마
sample = np.array([[25,150]])
pred = knn.predict(sample)
dist, idx = knn.kneighbors(sample)

# 데이터 정규화
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
sample_scaled = (sample - mean) / std

# 정규화 후 KNN
knn.fit(X_train_scaled, y_train)
dist_scaled, idx_scaled = knn.kneighbors(sample_scaled)

# 시각화
plt.figure(figsize=(10,6))

# 원본 데이터
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='#1f77b4', s=80, label='도미')
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], c='#ff7f0e', s=80, label='빙어')

# 원본 최근접 이웃
plt.scatter(X_train[idx[0],0], X_train[idx[0],1], marker='s', c='green', s=100, label='최근접 이웃')

# 수상한 도마
plt.scatter(sample[0,0], sample[0,1], marker='^', c='red', s=120, label='수상한 도마')

# 정규화 데이터(투명하게)
plt.scatter(X_train_scaled[y_train==1,0], X_train_scaled[y_train==1,1], c='#1f77b4', alpha=0.2, s=80)
plt.scatter(X_train_scaled[y_train==0,0], X_train_scaled[y_train==0,1], c='#ff7f0e', alpha=0.2, s=80)
plt.scatter(X_train_scaled[idx_scaled[0],0], X_train_scaled[idx_scaled[0],1], marker='s', c='green', alpha=0.2, s=100)
plt.scatter(sample_scaled[0,0], sample_scaled[0,1], marker='^', c='red', alpha=0.2, s=120)

plt.xlabel('길이')
plt.ylabel('무게')
plt.title('KNN 시각화: 원본 데이터 + 정규화 + 최근접 이웃')
plt.legend()
plt.grid(True)
plt.show()
