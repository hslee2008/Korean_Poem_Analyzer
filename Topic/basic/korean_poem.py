import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load each dataset
df1 = pd.read_csv('../src/자연에 대한 향수.csv')
df2 = pd.read_csv('../src/민중의 끈질긴 생명력.csv')

# Load the dataset
df = pd.concat([df1, df2])

# Split the dataset into input (X) and output (y) variables
X = df['본문']
y_year = df['연도']
y_topic = df['주제']

# Preprocess the data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model_year = MultinomialNB()
model_year.fit(X_vectorized, y_year)

modal_topic = MultinomialNB()
modal_topic.fit(X_vectorized, y_topic)

# Function to predict the author and year of a new poem
def predict_author_and_year(poem_text):
    poem_vectorized = vectorizer.transform([poem_text])
    predicted_year = model_year.predict(poem_vectorized)[0]
    predicted_topic = modal_topic.predict(poem_vectorized)[0]
    return predicted_year, predicted_topic

# Example usage
new_poem = """벼는 서로 어우러져

기대고 산다.

햇살이 따가워질수록

깊이 익어 스스로를 아끼고

이웃들에게 저를 맡긴다.

 

서로가 서로의 몸을 묶어

더 튼튼해진 백성들을 보아라.

죄도 없이 죄지어서 더욱 불타는

마음들을 보아라. 벼가 춤출 때,

벼는 소리 없이 떠나간다.

 

벼는 가을 하늘에도

서러운 눈 씻어 맑게 다스릴 줄 알고

바람 한 점에도

제 몸의 노여움을 덮는다.

저의 가슴도 더운 줄을 안다.

 

벼가 떠나가며 바치는

이 넓디넓은 사랑,

쓰러지고 쓰러지고 다시 일어서서 드리는

이 피 묻은 그리움,

이 넉넉한 힘……"""
predicted_year, predicted_topic = predict_author_and_year(new_poem)
print(f"The predicted year of the new poem is: {predicted_year}")
print(f"The predicted topic of the new poem is: {predicted_topic}")
