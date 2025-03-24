# Baes
# Bayes Teoremasi va NLPda Naïve Bayes Klassifikatori

Bu repository Bayes teoremasi va uning NLP (Natural Language Processing) dagi qo‘llanilishi bo‘yicha tushunchalar, nazariy asoslar va Python kodini o‘z ichiga oladi.

## 📌 Bayes Teoremasi

Bayes teoremasi shartli ehtimollarni hisoblash uchun ishlatiladi:

\(P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}\)

Bu yerda:

- **P(A)** - Hodisa A ning boshlang‘ich ehtimoli (Prior Probability)
- **P(B | A)** - A bo‘lgan taqdirda B ning ehtimoli (Likelihood)
- **P(B)** - B hodisasining umumiy ehtimoli (Evidence)
- **P(A | B)** - B mavjud bo‘lganda A ning yangilangan ehtimoli (Posterior Probability)

## 📌 NLPda Naïve Bayes Klassifikatori

Naïve Bayes klassifikatori matn klassifikatsiyasi (masalan, spam filtratsiyasi, hissiyotlarni aniqlash) uchun ishlatiladi.

1. **Matnni tozalash**: So‘zlarni kichik harfga o‘tkazish va punktuatsiyani olib tashlash
2. **Tokenizatsiya**: Matnni alohida so‘zlarga ajratish
3. **Vectorization (Bag of Words)**: Matnni raqamli vektor shakliga o‘tkazish
4. **Klassifikatsiya**: Har bir klass ehtimoli hisoblanadi va eng yuqori ehtimolga ega bo‘lgan klass tanlanadi

## 📌 Python Implementatsiyasi

Quyidagi kod orqali oddiy spam detektor yaratish mumkin:

```python
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Ma'lumotlarni tayyorlash
data = [
    ("Free money now!!!", "spam"),
    ("Win a lottery prize", "spam"),
    ("Call me when you are free", "ham"),
    ("Meeting at 5 PM", "ham"),
    ("Get 100% discount now", "spam"),
    ("Are you available tomorrow?", "ham"),
    ("Congratulations! You won a prize", "spam"),
    ("Let's go to the party tonight", "ham")
]

df = pd.DataFrame(data, columns=["text", "label"])
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# 2. Matnni tozalash
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)

# 3. Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 4. Train/Test ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelni o‘rgatish
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Test qilish
y_pred = model.predict(X_test)
print("Aniqlik:", accuracy_score(y_test, y_pred))

# 7. Yangi matnni tekshirish
new_message = ["Get a free lottery ticket now"]
new_message_cleaned = [clean_text(msg) for msg in new_message]
new_message_vectorized = vectorizer.transform(new_message_cleaned)
prediction = model.predict(new_message_vectorized)
print("Spam" if prediction[0] == 1 else "Ham (Oddiy xabar)")
```

## 📌 Ishga tushirish

1. **Python va kerakli kutubxonalarni o‘rnatish**
   ```bash
   pip install numpy pandas scikit-learn
   ```
2. **Scriptni ishga tushirish**
   ```bash
   python script.py
   ```
3. **Model natijalarini ko‘rish**
   - Model aniqlik darajasini chiqaradi
   - Yangi matnlar uchun spam yoki oddiy xabar ekanligini aniqlaydi

## 📌 Xulosa

- Bayes teoremasi orqali shartli ehtimollarni yangilash mumkin
- NLPda Naïve Bayes sodda va samarali matn klassifikatori hisoblanadi
- Python yordamida spam detektor yaratish oson va tez amalga oshiriladi

✅ Agar ushbu loyiha sizga foydali bo‘lsa, ⭐ yulduzcha bosing va loyihani baham ko‘ring!

