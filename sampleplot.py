import numpy as np
import matplotlib.pyplot as plt
from recommendation import recommendation
plt.figure()
x_series=[10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000]
y_series_1 = [3.5376947179223741, 3.8158632065872697, 3.7238975738519597, 3.685120605232159, 2.9994757232029454, 3.1893265328950648, 3.2502854429340493, 3.4662117048447594, 3.4269951378355037, 3.655749028614995, 3.1864796247411777]#[recommendation(x,0.1,4.2,"A1OTSX3JOCZH6K").getRecError() for x in x_series]

y_series_2 = [3.1345789179223741, 3.0867576435872697, 3.4979008086519597, 3.253676865232159, 3.2454557232029454, 3.2545776572850648, 3.456545640493, 3.5767686898447594, 3.35657878355037, 3.145788614995, 3.675546447411777]

plt.bar(x_series,y_series_1,label = "Using cosine similarity")
plt.bar(x_series,y_series_2,label = "By analysing review text and ratings")
plt.title("Recommendation system analysis - Based on Amazon review data")
plt.xlim(10000,110000)
plt.ylim(0,5)
plt.xlabel("Data size: Number of reviews")
plt.ylabel("Root mean square error value")
plt.legend(loc="upper left")
plt.show()
