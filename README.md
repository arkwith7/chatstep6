# 1. Tensorflow2.x로 Upgrade
## 1.1. 텐서플로우 2.x 가상환경을 생성

```
conda create -n tf2_env python=3.6
```

## 1.2 한글처리기 설치

```
pip install JPype1-0.7.1-cp36-cp36m-win_amd64.whl
pip install konlpy
```

## 1.3 Chatting Application을 위한 Python Package 설치
장고 어플리케이션이 있는 디렉토리(requirements.txt가 존재하는..)로 이동하여,
```
pip install -r requirements.txt
```
## 1.4 텐서플로우설치 및 케라스 설치

```
pip install tensorflow
pip install keras
```
## 1.5 가상환경에 주피터노트북 설치

```
pip install ipykernel
python -m ipykernel install --user --name tf2_env --display-name "tf2_env"
```

# 2. 변경 파일
## 2.1 챗봇엔진 학습 예측 부분 변경
**~\mychatsite\chatapp\ArkChatFramework\DialogManager\tf_learning_model.py**
```
    # restore all of trained tflearn model file
    def restore_training_model(self):
        """
        Tensorflow 딥러닝으로 생성한 자연어 이해 모델을 읽어들여서 반환한다.
        """
        if not (self.train_x and self.train_y):
            self.restore_training_data_structures()

        # tensorflow2.x로 업그레이드(2020.09.18)
        model = tf.keras.models.load_model(self.tflearn_model_file)

        self.tf_learning_model = model
...........
.............
...............
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def get_bag_of_words(self, sentence, show_details=False):

        # tokenize the pattern
        sentence_words = self.get_words_from_sentence(sentence)

        # bag of words
        if not self.dialog_words:
            self.restore_training_data_structures()
        
        info_bag_of_words = []
        bag = [0]*len(self.dialog_words)  
        for s in sentence_words:
            for i,w in enumerate(self.dialog_words):
                if w == s: 
                    bag[i] = 1
                    info_bag_of_words.append(w)
                    
        if show_details:
            self.logger.debug("found in bag: %s" % info_bag_of_words)
    
        return(bag)
..........
.............
..............
class NLULearning(LearningModel):
    '''
    Tensorflow 딥러닝으로 자연어 이해 모델을 생성한 한다.
    '''
    def __init__(self, language, learning_model_files):
        '''
        NLULearning Constructor
        '''           
.........
..........
...........
    def create_tensorflow_learning_model(self):
        """
        딥러닝(tensorflow)을 통한 자연어 이해 모델 생성
        """
        if not (self.train_x and self.train_y):
            self.prepare_machine_learning()
        
        # Using Keras, Build neural network(tensorflow2.x로 2020.09.18 변경)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(len(self.train_x[0]),)),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Dense(len(self.train_y[0]), activation="softmax"),
            ])
        
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(self.train_x, self.train_y, epochs=1000, batch_size=8)    
        # save the trained model to directory
        model.save(self.tflearn_model_file)
        self.logger.debug("create_tensorflow_learning_model() success... : %s" % self.tflearn_model_file)

        self.tf_learning_model = model
.........
..........
...........
    

```
