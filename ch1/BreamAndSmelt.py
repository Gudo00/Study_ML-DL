# 과학 계산용 그래프 패키지(Matplotlib), k-최근접 이웃 알고리즘 클래스(KNeighborsClassifier)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 도미, 빙어 그래프 그리기
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('Length')
plt.ylabel('Weight')

"""## 첫 번째 머신러닝 프로그램"""
# 두 리스트를 하나로 합침
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 사이킷런을 사용하기 때문에 2차원 리스트로 만들어야 함. zip() 사용으로 length, weight 리스트 생성
fish_data = [[l,w] for l,w in zip(length,weight)]

plt.show()

print('🐟 생선 데이터 🐠')
print(fish_data)
print('-------------------------------------')

# 도미 = 1, 빙어 = 0
fish_target = [1] * 35 + [0] * 14
print('🐟 도미, 빙어 리스트 (도미 = 1, 빙어 = 0)')
print(fish_target)
print('-------------------------------------')

# fish_data와 fish_target을 전달 후 도미를 찾기 위한 기준을 학습
# 훈련 객체 생성
kn = KNeighborsClassifier()

# fit() 메서드로 알고리즘 훈련
kn.fit(fish_data, fish_target)

# score() 메서드로 모델 평가
print('🐟 모델 평가')
print(kn.score(fish_data, fish_target))
print('-------------------------------------')

"""## K-최근접 이웃 알고리즘"""
# predict() 메서드로 새로운 데이터(30, 600)의 정답을 예측.
print('🔥 K-최근접 이웃 알고리즘')
print('🐟 새로운 값 예측.(도미 = 1, 빙어 = 0) x = 30, y = 600')
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30,600, color='red', marker='x')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()
print(kn.predict([[30, 600]]))
print('-------------------------------------')

print('🔥 데이터 확인 !')
# KNeighborsClassifier 클래스의 _fix_X 속성에 fish_data, _y 속성에 fish_target
print('🐟생선 데이터(fish_data)')
print(kn._fit_X)
print('-------------------------------------')
print('🐟도미, 빙어 리스트')
print(kn._y)
print('-------------------------------------')

# 참고 데이터를 49개로 한 모델 (기본 값은 5)
kn49 = KNeighborsClassifier(n_neighbors=49)

# 하지만 가장 가까운 데이터 49개를 사용하는 k-최근접 이웃 모델에 fish_data를 적용하고, 도미가 35개이기 때문에.. 무조건 도미로 예측함
print('🔥 데이터 49개 적용한 K-최근접 이웃 알고리즘')
kn49.fit(fish_data, fish_target)
print(kn49.score(fish_data, fish_target))

