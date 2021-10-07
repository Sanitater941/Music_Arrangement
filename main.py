import music21
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


#작은별 악보를 ABC 표기로 표현
little_star="tinynotation: 4/4 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2 g4 g4 f4 f4 e4 e4 g2 g4 g4 f4 f4 e4 e4 d2 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2"

#봄나들이 악보를 ABC 표기로 표현
spring_picnic="tinynotation: 4/8 g8 e8 g8 e8 g8 a8 g4 e8 g8 e8 c8 d8 e8 c4 g8 e8 g8 e8 g8 a8 g4 b8 a8 g8 e8 d8 e8 c4"

#나비야 나비야 악보를 ABC 표기로 표현
butterfly="tinynotation: 2/4 g8 e8 e4 f8 d8 d4 c8 d8 e8 f8 g8 g8 g4 g8 e8 e8 e8 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4 d8 d8 d8 d8 e8 f4 e8 e8 e8 e8 e8 f8 g4 g8 e8 e4 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4"


#계이름과 숫자를 상호 변환하는 표(딕셔너리 자료구조를 사용함)
note2num={'c':1, 'd':2, 'e':3, 'f':4, 'g':5, 'a':6, 'b':7}
num2note={1:'c', 2:'d', 3:'e', 4:'f', 5:'g', 6:'a', 7:'b'}

#ABC 표기를 시계열 데이터로 변환
def abc2timeseries(s):
    notes=s.split(' ')[2:]
    seq=[]
    for i in notes:
        seq.append([note2num[i[0]],int(i[1])])
    return seq

#시계열 데이터를 ABC 표기로 변환
def timeseries2abc(t):
    s='tinynotation: 4/4'
    for i in t:
        s=s+' '+num2note[i[0]]+str(i[1])
    return s

#원핫 코드로 변환하는 표
onehot=[[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[1,4],[2,4],[3,4],[4,4],[5,4]
        ,[6,4],[7,4],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8]]

#레이블을 원핫 코드로 변환
def to_onehot(l):
    t=[]
    for i in range(len(l)):
        a=np.zeros(len(onehot))
        a[onehot.index(list(l[i]))]=1.0
        t.append(a)
    return np.array(t)

#시계열 데이터를 훈련 집합으로 자름
def seq2dataset(seq,window,horizon):
    X=[];Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

w=8 #윈도우 크기
h=1 #수평선 계수

#세 곡을 시계열로 변환하고 결합
seq1=abc2timeseries(little_star)
seq2=abc2timeseries(butterfly)
seq3=abc2timeseries(spring_picnic)
seq = seq1 + seq2 +seq3

X,Y=seq2dataset(seq, w, h)
print(X.shape,Y.shape)
print(X[0],Y[0])

#훈련집합 구축
split=int(len(X)*1.0)
x_train = X[0:split]; y_train=Y[0:split]
y_train=to_onehot(y_train)

#LSTM 모델 설계와 학습
model=Sequential()
model.add(LSTM(units=128,activation='relu',input_shape=x_train[0].shape))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=1, verbose=2)

#학습된 모델로 편곡을 하는 함수(fisrt_measure: 첫 소절, duration: 생성될 곡의 길이)
def arranging_music(model, first_measure,duration):
    music = first_measure
    for i in range(duration):
        p=model.predict(np.float32(np.expand_dims(music[-w:],axis=0)))
        music=np.append(music,[onehot[np.argmax(p)]],axis=0)
    return timeseries2abc(music)

x_train[0]=[[5,4],[3,4],[3,2],[4,4],[2,4],[2,2],[1,4],[2,4]]
new_song = arranging_music(model, x_train[0], 50)

print(new_song)
music21.converter.parse(new_song).show('mid')

