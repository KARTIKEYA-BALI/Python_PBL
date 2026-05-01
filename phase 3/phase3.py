import joblib
import nltk
import regex as re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



def listing(s):

    if type(s)==list:
        return s
    

    text = re.sub(r'[^a-zA-Z\s]', '', s)
    stp_wrd = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    ps = PorterStemmer()
    
    return [ps.stem(i) for i in tokens if i not in stp_wrd]

tfidf = joblib.load(r"C:\Users\balih\Desktop\AI ML lab work\Python_PBL\phase 3\tfidf.pkl")
model = joblib.load(r"C:\Users\balih\Desktop\AI ML lab work\Python_PBL\phase 3\model.pkl")


s = input("enter your email: ")


x = tfidf.transform([s])
k = model.predict(x)

if k[0] == 1:
    print("\n\n\nTHE EMAIL MIGHT BE SPAM")
else:
    print("\n\n\nTHE EMAIL MIGHT NOT BE SPAM")
