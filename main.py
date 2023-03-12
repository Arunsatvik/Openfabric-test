# Importing required libraries
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
import random
import warnings

# Ignore warnings for clear output view
warnings.filterwarnings('ignore')

# Read the text file which has information on science
f = open('/Users/SATVIK/Desktop/openfabric-test/science.txt', 'r', errors='ignore')
raw_doc = f.read()

# Convert entire text to lowercase
raw_doc = raw_doc.lower()

# Download required data from the nltk library
nltk.download('punkt')  # Using the Punkt tokenizer
nltk.download('wordnet')  # Using the wordnet dictionary
nltk.download('omw-1.4')

# Tokenize the text into sentences and words
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

# Initialize lemmatizer for stemming words
lemmer = nltk.stem.WordNetLemmatizer()


# Define function for lemmatizing tokens
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# Create dictionary for removing punctuation
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)


# Define function for normalizing and lemmatizing text
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


# Define greetings inputs and responses
greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
greet_responses = ('hi', 'Hey', 'Hey There!', 'There there!!')


# Define function for responding to greetings
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)


# Define function for generating response to user input
def response(user_response):
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response + "I am sorry. Unable to understand you!"
        return robo1_response
    else:
        robo1_response = robo1_response + sentence_tokens[idx]
        return robo1_response


def config(configuration: ConfigClass):
    # TODO Add code here
    pass


# Define function for executing the chatbot
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    global word_tokens
    output = []

    for text in request.text:
        print("\n" + "User: " + text)
        user_response = text
        user_response = user_response.lower()
        if user_response != 'bye':
            if user_response == 'thank you' or user_response == 'thanks':
                print('Bot: You are Welcome..')
            else:
                if greet(user_response) is not None:
                    print('Bot: ' + greet(user_response))
                else:
                    sentence_tokens.append(user_response)
                    word_tokens = word_tokens + nltk.word_tokenize(user_response)
                    list(set(word_tokens))
                    print('Bot: ', end='')
                    l = response(user_response) + "\n"
                    print(l)
                    sentence_tokens.remove(user_response)
                    output.append(l)
        else:
            print('Bot: Goodbye!')
    return SimpleText(dict(text=output))


# using the simple text, pass the preloaded questions to the execute function
t = SimpleText({'text': ['hi',
                         'Why Science has unquestionably been the most successful human endeavor in the history of civilization?',
                         'Another name for empirical evidence?', 'who all can think like scientist?', 'thanks']})
k = OpenfabricExecutionRay(1)
execute(t, k)

# instead of using simple text, user input
flag = True
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            flag = False
            print('Bot: You are Welcome..')
        else:
            if greet(user_response) is not None:
                print('Bot: ' + greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot: ', end='')
                print(response(user_response) + "\n")
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Goodbye!')
