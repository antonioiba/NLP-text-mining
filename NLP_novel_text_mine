#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:21:30 2017

@author: mulugetasemework, thank you for NLP primer by bonzanini and WordCloud
 by amueller
 This code text-mines a ascii-saved text version of the book: "Love in another country" and gives a wordcloud
 png of the most frequent words, both in regular figure and mask background 
 format. 
 It also gives a frequency-sorted version of the important lists (counters) for
 each category
"""

import os
maxFontSize = 100
maxWords = 1000
os.chdir('/Users/mulugetasemework/Documents/Python/NLP')
os.getcwd()
allText =  open('LoveInAnotherCountry.txt','r') 
 
allText = allText.read()

#Tokenisation -----------------------------------------------------------------
from nltk.tokenize import word_tokenize

try:  # py3
    compTokens = [t for t in word_tokenize(allText)]
except UnicodeDecodeError:  # py27
    compTokens = [t for t in word_tokenize(allText.decode('utf-8'))]

#Counting Words ---------------------------------------------------------------

from collections import Counter

totTerm_frequency = Counter(compTokens)
totTerm_frequencySorted = totTerm_frequency.most_common()
for word, freq in totTerm_frequency.most_common(20):
    print("{}\t{}".format(word, freq))


document_frequency = Counter()

tokens = word_tokenize(allText)
unique_tokens = set(tokens)
document_frequency.update(unique_tokens)


#Stop-words-------------------------------------------------------------------
from nltk.corpus import stopwords
import string

print(stopwords.words('english'))
print(len(stopwords.words('english')))
print(string.punctuation)


stop_list = stopwords.words('english') + list(string.punctuation)

tkns_NoStop = [token for token in compTokens
                        if token not in stop_list]

totTerm_freqNoStop = Counter(tkns_NoStop)
totTerm_freqNoStopSorted = Counter(tkns_NoStop).most_common()

    
#Text Normalisation ----------------------------------------------------------

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
comptkns_lower = [t.lower() for t in compTokens]
tkns_normalised = [stemmer.stem(t) for t in comptkns_lower
                                     if t not in stop_list]

totTerm_freqNorm = Counter(tkns_normalised)
totTerm_freqNormSorted = Counter(tkns_normalised).most_common()


# n-grams--------------------------------------------------------------------


from nltk import ngrams

phrases2ng = Counter(ngrams(comptkns_lower, 2))
phrases2ngSorted = Counter(ngrams(comptkns_lower, 2)).most_common()

phrases3ng = Counter(ngrams(comptkns_lower, 3))
phrases3ngSorted = Counter(ngrams(comptkns_lower, 3)).most_common()

# n-grams and stop-words -----------------------------------------------------
phrases3ngNs = Counter(ngrams(tkns_NoStop, 2))
phrases3ngNsSorted = Counter(ngrams(tkns_NoStop, 2)).most_common()


phrases3ngNs = Counter(ngrams(tkns_NoStop, 3))
phrases3ngNsSorted = Counter(ngrams(tkns_NoStop, 3)).most_common()

# Word cloud -----------------------------------------------------------------

from wordcloud import WordCloud
import matplotlib.pyplot as plt


from scipy import misc
face = misc.face()
aliceMask = misc.imread('alice.jpg')
heartMask = misc.imread('broken-heart.gif_595.jpg')
starMask = misc.imread('star5.jpg')
handEyeMask = misc.imread('hearthandeyehand.jpg')
quotesMask = misc.imread('quotes.jpg')


#ax1
wordcloud0 = WordCloud(max_font_size= maxFontSize,max_words=maxWords).generate(str(allText))
wc = WordCloud(background_color="black",  mask=aliceMask,max_words=maxWords)
#ax3
wordcloud00 = wc.generate(allText)

#ax2
wordcloud1 = WordCloud(max_font_size=maxFontSize,max_words=maxWords).generate_from_frequencies(totTerm_frequency.items())
wc = WordCloud(background_color="black",  mask=heartMask,max_words=maxWords)
#ax4
wordcloud11 = wc.generate_from_frequencies(totTerm_frequency.items())

#ax5
wordcloud2 = WordCloud(max_font_size= maxFontSize,max_words=maxWords).generate_from_frequencies((totTerm_freqNoStop.items()))
wc = WordCloud(background_color="white",  mask=starMask,max_words=maxWords)
#ax7
wordcloud22 = wc.generate_from_frequencies(totTerm_freqNoStop.items())
#ax6
wordcloud3 = WordCloud(max_font_size= maxFontSize,max_words=maxWords).generate_from_frequencies((totTerm_freqNorm.items()))
wc = WordCloud(background_color="white",  mask=quotesMask,max_words=maxWords)
#ax8
wordcloud33 = wc.generate_from_frequencies(totTerm_freqNorm.items())


plt.close("all")
plt.figure(figsize=(10,30))
plt.axis("off")
f, ((ax1, ax2),(ax3, ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4, 2)

f.subplots_adjust(left=0.06, right=0.99, top=0.75, bottom=0.05, hspace=0.28, wspace= 0.01)

#-------------------------------------------------------------------
ax1.imshow(wordcloud0, interpolation="bilinear")
ax1.axis("off")
ax1.set_title('All text',fontsize=10)

#-------------------------------------------------------------------
ax3.imshow(wordcloud00, interpolation="bilinear")
ax3.axis("off")


#---------
ax2.imshow(wordcloud1, interpolation="bilinear")
ax2.axis("off")
ax2.set_title('All tokens',fontsize=10)

#---------
ax4.imshow(wordcloud11, interpolation="bilinear")
ax4.axis("off")

#--------
ax5.imshow(wordcloud2, interpolation="bilinear")
ax5.axis("off")
ax5.set_title('All tokens, no stop words',fontsize=10)
#---------
ax7.imshow(wordcloud22, interpolation="bilinear")
ax7.axis("off")

#---------
ax6.imshow(wordcloud3, interpolation="bilinear")
ax6.axis("off")
ax6.set_title('All tokens, canonized (normalized)',fontsize=10)
#---------
ax8.imshow(wordcloud33, interpolation="bilinear")
ax8.axis("off")

figTitle = 'The top ' + str(maxWords) +' Most frequent words/characters \n in the novel: "Love in Another Country" \n'
titleLink =   "www.python.org"
plt.suptitle(figTitle,fontsize=15)
plt.show()
f.canvas.manager.window.activateWindow()
f.canvas.manager.window.raise_()
f.savefig('test.png')




