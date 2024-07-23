import string
import nltk
import xlsxwriter
import collections

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from xlsxwriter import Workbook


def tokenize():
    # Tokenizing
    print("Enter a sentence")
    sentence = input()

    tokens = nltk.tokenize.word_tokenize(sentence)
    # print(tokens)

    workbook = xlsxwriter.Workbook('arrays.xlsx')
    worksheet = workbook.add_worksheet()

    array = [tokens]

    row = 0

    for col, data in enumerate(array):
        worksheet.write_column(row, col, data)

    workbook.close()


def stop_word():
    # StopWord
    print("Enter a sentence")
    sentence = input()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(sentence)
    list_stop_word = set(stopwords.words('indonesian'))

    removed = []
    for t in tokens:
        if t not in list_stop_word:
            removed.append(t)

    # print(removed)

    workbook = xlsxwriter.Workbook('arrays.xlsx')
    worksheet = workbook.add_worksheet()

    array = [removed]

    row = 0

    for col, data in enumerate(array):
        worksheet.write_column(row, col, data)

    workbook.close()


def stemming():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    print("Enter a sentence")
    sentences = input()
    sentence = stemmer.stem(sentences)

    print(sentence)


def count_word():
    global word
    print("Enter a sentence")
    sentences = input()

    li = sentences.split(' ')
    get = collections.Counter(li).most_common()

    for word in get:
        print("%s\t: %d" % (word[0], word[1]))

    workbook: Workbook = xlsxwriter.Workbook('arrays.xlsx')
    worksheet = workbook.add_worksheet()

    array = ["%s\t: %d" % (word[0], word[1])]

    row = 0

    for col, data in enumerate(array):
        worksheet.write_column(row, col, data)

    workbook.close()


print("1 for tokenizing\n"
      "2 for stop word\n"
      "3 for stemming\n"
      "4 for count word")
print("Choose number:")
number = input()
if number == "1":
    tokenize()
elif number == "2":
    stop_word()
elif number == "3":
    stemming()
elif number == "4":
    count_word()
else:
    print("NOPE")
