import re

import matplotlib
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

x = """
total = 3937238 	Query 1a >> 		 Runtime: 2 ms - Result correct: true
total = 3433384 	Query 1b >> 		 Runtime: 1 ms - Result correct: true
total = 1789519 	Query 1c >> 		 Runtime: 2 ms - Result correct: true
total = 4098578 	Query 1d >> 		 Runtime: 2 ms - Result correct: true
total = 9671147 	Query 2a >> 		 Runtime: 6 ms - Result correct: true
total = 9663712 	Query 2b >> 		 Runtime: 5 ms - Result correct: true
total = 9661373 	Query 2c >> 		 Runtime: 3 ms - Result correct: true
total = 9746215 	Query 2d >> 		 Runtime: 17 ms - Result correct: true
total = 5690777 	Query 3a >> 		 Runtime: 3 ms - Result correct: true
total = 4917913 	Query 3b >> 		 Runtime: 2 ms - Result correct: true
total = 6979666 	Query 3c >> 		 Runtime: 3 ms - Result correct: true
total = 6054563 	Query 4a >> 		 Runtime: 4 ms - Result correct: true
total = 4923949 	Query 4b >> 		 Runtime: 2 ms - Result correct: true
total = 7079358 	Query 4c >> 		 Runtime: 4 ms - Result correct: true
total = 1190956 	Query 5a >> 		 Runtime: 0 ms - Result correct: true
total = 945975 	    Query 5b >> 		 Runtime: 0 ms - Result correct: true
total = 2751448 	Query 5c >> 		 Runtime: 1 ms - Result correct: true
total = 41159943 	Query 6a >> 		 Runtime: 7 ms - Result correct: true
total = 40768722 	Query 6b >> 		 Runtime: 10 ms - Result correct: true
total = 40768715 	Query 6c >> 		 Runtime: 7 ms - Result correct: true
total = 42149737 	Query 6d >> 		 Runtime: 9 ms - Result correct: true
total = 42149730 	Query 6e >> 		 Runtime: 7 ms - Result correct: true
total = 46317226 	Query 6f >> 		 Runtime: 48 ms - Result correct: true
total = 37832576 	Query 7a >> 		 Runtime: 32 ms - Result correct: true
total = 37117140 	Query 7b >> 		 Runtime: 23 ms - Result correct: true
total = 39107842 	Query 7c >> 		 Runtime: 141 ms - Result correct: true
total = 3599311 	Query 8a >> 		 Runtime: 4 ms - Result correct: true
total = 1030678 	Query 8b >> 		 Runtime: 1 ms - Result correct: true
total = 46535463 	Query 8c >> 		 Runtime: 100 ms - Result correct: true
total = 46535463 	Query 8d >> 		 Runtime: 85 ms - Result correct: true
total = 6699653 	Query 9a >> 		 Runtime: 3 ms - Result correct: true
total = 5557107 	Query 9b >> 		 Runtime: 2 ms - Result correct: true
total = 10181455 	Query 9c >> 		 Runtime: 40 ms - Result correct: true
total = 11092999 	Query 9d >> 		 Runtime: 65 ms - Result correct: true
total = 6796042 	Query 10a >> 		 Runtime: 3 ms - Result correct: true
total = 7577939 	Query 10b >> 		 Runtime: 15 ms - Result correct: true
total = 9018798 	Query 10c >> 		 Runtime: 128 ms - Result correct: true
total = 6784803 	Query 11a >> 		 Runtime: 4 ms - Result correct: true
total = 5874257 	Query 11b >> 		 Runtime: 2 ms - Result correct: true
total = 8178451 	Query 11c >> 		 Runtime: 4 ms - Result correct: true
total = 8388393 	Query 11d >> 		 Runtime: 7 ms - Result correct: true
total = 3467108 	Query 12a >> 		 Runtime: 3 ms - Result correct: true
total = 18912267 	Query 12b >> 		 Runtime: 7 ms - Result correct: true
total = 4256927 	Query 12c >> 		 Runtime: 8 ms - Result correct: true
total = 21362975 	Query 13a >> 		 Runtime: 13 ms - Result correct: true
total = 18911949 	Query 13b >> 		 Runtime: 14 ms - Result correct: true
total = 18910025 	Query 13c >> 		 Runtime: 6 ms - Result correct: true
total = 21438043 	Query 13d >> 		 Runtime: 142 ms - Result correct: true
total = 6959910 	Query 14a >> 		 Runtime: 5 ms - Result correct: true
total = 5604472 	Query 14b >> 		 Runtime: 3 ms - Result correct: true
total = 7590701 	Query 14c >> 		 Runtime: 5 ms - Result correct: true
total = 6549308 	Query 15a >> 		 Runtime: 3 ms - Result correct: true
total = 5799272 	Query 15b >> 		 Runtime: 2 ms - Result correct: true
total = 9464364 	Query 15c >> 		 Runtime: 4 ms - Result correct: true
total = 9473391 	Query 15d >> 		 Runtime: 5 ms - Result correct: true
total = 48599326 	Query 16a >> 		 Runtime: 14 ms - Result correct: true
total = 51059393 	Query 16b >> 		 Runtime: 96 ms - Result correct: true
total = 49477987 	Query 16c >> 		 Runtime: 27 ms - Result correct: true
total = 49223637 	Query 16d >> 		 Runtime: 28 ms - Result correct: true
total = 46333958 	Query 17a >> 		 Runtime: 22 ms - Result correct: true
total = 46176887 	Query 17b >> 		 Runtime: 11 ms - Result correct: true
total = 46143377 	Query 17c >> 		 Runtime: 8 ms - Result correct: true
total = 46145929 	Query 17d >> 		 Runtime: 9 ms - Result correct: true
total = 50158050 	Query 17e >> 		 Runtime: 45 ms - Result correct: true
total = 46677429 	Query 17f >> 		 Runtime: 59 ms - Result correct: true
total = 21132402 	Query 18a >> 		 Runtime: 8 ms - Result correct: true
total = 3116437 	Query 18b >> 		 Runtime: 2 ms - Result correct: true
total = 7081615 	Query 18c >> 		 Runtime: 16 ms - Result correct: true
total = 6617426 	Query 19a >> 		 Runtime: 3 ms - Result correct: true
total = 5113768 	Query 19b >> 		 Runtime: 2 ms - Result correct: true
total = 9485701 	Query 19c >> 		 Runtime: 4 ms - Result correct: true
total = 24781861 	Query 19d >> 		 Runtime: 95 ms - Result correct: true
total = 47358164 	Query 20a >> 		 Runtime: 9 ms - Result correct: true
total = 42284857 	Query 20b >> 		 Runtime: 8 ms - Result correct: true
total = 46632470 	Query 20c >> 		 Runtime: 14 ms - Result correct: true
total = 6938700 	Query 21a >> 		 Runtime: 4 ms - Result correct: true
total = 7033581 	Query 21b >> 		 Runtime: 4 ms - Result correct: true
total = 8710813 	Query 21c >> 		 Runtime: 5 ms - Result correct: true
total = 7457803 	Query 22a >> 		 Runtime: 7 ms - Result correct: true
total = 7329107 	Query 22b >> 		 Runtime: 7 ms - Result correct: true
total = 8020206 	Query 22c >> 		 Runtime: 9 ms - Result correct: true
total = 10326064 	Query 22d >> 		 Runtime: 10 ms - Result correct: true
total = 8870401 	Query 23a >> 		 Runtime: 3 ms - Result correct: true
total = 8736223 	Query 23b >> 		 Runtime: 3 ms - Result correct: true
total = 9237983 	Query 23c >> 		 Runtime: 3 ms - Result correct: true
total = 12869990 	Query 24a >> 		 Runtime: 6 ms - Result correct: true
total = 12393490 	Query 24b >> 		 Runtime: 37 ms - Result correct: true
total = 11447380 	Query 25a >> 		 Runtime: 7 ms - Result correct: true
total = 8919141 	Query 25b >> 		 Runtime: 4 ms - Result correct: true
total = 11605552 	Query 25c >> 		 Runtime: 8 ms - Result correct: true
total = 46825310 	Query 26a >> 		 Runtime: 18 ms - Result correct: true
total = 46316561 	Query 26b >> 		 Runtime: 13 ms - Result correct: true
total = 48012506 	Query 26c >> 		 Runtime: 22 ms - Result correct: true
total = 7054131 	Query 27a >> 		 Runtime: 7 ms - Result correct: true
total = 6190133 	Query 27b >> 		 Runtime: 4 ms - Result correct: true
total = 8845902 	Query 27c >> 		 Runtime: 4 ms - Result correct: true
total = 8523829 	Query 28a >> 		 Runtime: 8 ms - Result correct: true
total = 6543958 	Query 28b >> 		 Runtime: 5 ms - Result correct: true
total = 8155294 	Query 28c >> 		 Runtime: 5 ms - Result correct: true
total = 12555821 	Query 29a >> 		 Runtime: 7 ms - Result correct: true
total = 12492399 	Query 29b >> 		 Runtime: 11 ms - Result correct: true
total = 16769732 	Query 29c >> 		 Runtime: 14 ms - Result correct: true
total = 10477858 	Query 30a >> 		 Runtime: 7 ms - Result correct: true
total = 9096938 	Query 30b >> 		 Runtime: 6 ms - Result correct: true
total = 11740640 	Query 30c >> 		 Runtime: 9 ms - Result correct: true
total = 14098767 	Query 31a >> 		 Runtime: 7 ms - Result correct: true
total = 8969822 	Query 31b >> 		 Runtime: 4 ms - Result correct: true
total = 16642603 	Query 31c >> 		 Runtime: 8 ms - Result correct: true
total = 9610570 	Query 32a >> 		 Runtime: 2 ms - Result correct: true
total = 9610570 	Query 32b >> 		 Runtime: 4 ms - Result correct: true
total = 10597136 	Query 33a >> 		 Runtime: 7 ms - Result correct: true
total = 10188338 	Query 33b >> 		 Runtime: 6 ms - Result correct: true
total = 11247904 	Query 33c >> 		 Runtime: 16 ms - Result correct: true
"""

y_AMD = """
Query 1a >> 		 Runtime: 1204 ms - Result correct: true
Query 1b >> 		 Runtime: 1203 ms - Result correct: true
Query 1c >> 		 Runtime: 1203 ms - Result correct: true
Query 1d >> 		 Runtime: 1204 ms - Result correct: true
Query 2a >> 		 Runtime: 1209 ms - Result correct: true
Query 2b >> 		 Runtime: 1208 ms - Result correct: true
Query 2c >> 		 Runtime: 1206 ms - Result correct: true
Query 2d >> 		 Runtime: 1225 ms - Result correct: true
Query 3a >> 		 Runtime: 1204 ms - Result correct: true
Query 3b >> 		 Runtime: 1204 ms - Result correct: true
Query 3c >> 		 Runtime: 1206 ms - Result correct: true
Query 4a >> 		 Runtime: 1206 ms - Result correct: true
Query 4b >> 		 Runtime: 1204 ms - Result correct: true
Query 4c >> 		 Runtime: 1207 ms - Result correct: true
Query 5a >> 		 Runtime: 1201 ms - Result correct: true
Query 5b >> 		 Runtime: 1201 ms - Result correct: true
Query 5c >> 		 Runtime: 1203 ms - Result correct: true
Query 6a >> 		 Runtime: 1214 ms - Result correct: true
Query 6b >> 		 Runtime: 1216 ms - Result correct: true
Query 6c >> 		 Runtime: 1213 ms - Result correct: true
Query 6d >> 		 Runtime: 1217 ms - Result correct: true
Query 6e >> 		 Runtime: 1214 ms - Result correct: true
Query 6f >> 		 Runtime: 1263 ms - Result correct: true
Query 7a >> 		 Runtime: 1251 ms - Result correct: true
Query 7b >> 		 Runtime: 1229 ms - Result correct: true
Query 7c >> 		 Runtime: 1361 ms - Result correct: true
Query 8a >> 		 Runtime: 1206 ms - Result correct: true
Query 8b >> 		 Runtime: 1203 ms - Result correct: true
Query 8c >> 		 Runtime: 1312 ms - Result correct: true
Query 8d >> 		 Runtime: 1272 ms - Result correct: true
Query 9a >> 		 Runtime: 1204 ms - Result correct: true
Query 9b >> 		 Runtime: 1204 ms - Result correct: true
Query 9c >> 		 Runtime: 1243 ms - Result correct: true
Query 9d >> 		 Runtime: 1286 ms - Result correct: true
Query 10a >> 		 Runtime: 1206 ms - Result correct: true
Query 10b >> 		 Runtime: 1237 ms - Result correct: true
Query 10c >> 		 Runtime: 1260 ms - Result correct: true
Query 11a >> 		 Runtime: 1207 ms - Result correct: true
Query 11b >> 		 Runtime: 1204 ms - Result correct: true
Query 11c >> 		 Runtime: 1207 ms - Result correct: true
Query 11d >> 		 Runtime: 1208 ms - Result correct: true
Query 12a >> 		 Runtime: 1205 ms - Result correct: true
Query 12b >> 		 Runtime: 1210 ms - Result correct: true
Query 12c >> 		 Runtime: 1209 ms - Result correct: true
Query 13a >> 		 Runtime: 1239 ms - Result correct: true
Query 13b >> 		 Runtime: 1211 ms - Result correct: true
Query 13c >> 		 Runtime: 1212 ms - Result correct: true
Query 13d >> 		 Runtime: 1301 ms - Result correct: true
Query 14a >> 		 Runtime: 1208 ms - Result correct: true
Query 14b >> 		 Runtime: 1205 ms - Result correct: true
Query 14c >> 		 Runtime: 1209 ms - Result correct: true
Query 15a >> 		 Runtime: 1205 ms - Result correct: true
Query 15b >> 		 Runtime: 1204 ms - Result correct: true
Query 15c >> 		 Runtime: 1208 ms - Result correct: true
Query 15d >> 		 Runtime: 1208 ms - Result correct: true
Query 16a >> 		 Runtime: 1233 ms - Result correct: true
Query 16b >> 		 Runtime: 1298 ms - Result correct: true
Query 16c >> 		 Runtime: 1233 ms - Result correct: true
Query 16d >> 		 Runtime: 1223 ms - Result correct: true
Query 17a >> 		 Runtime: 1233 ms - Result correct: true
Query 17b >> 		 Runtime: 1220 ms - Result correct: true
Query 17c >> 		 Runtime: 1216 ms - Result correct: true
Query 17d >> 		 Runtime: 1216 ms - Result correct: true
Query 17e >> 		 Runtime: 1260 ms - Result correct: true
Query 17f >> 		 Runtime: 1265 ms - Result correct: true
Query 18a >> 		 Runtime: 1211 ms - Result correct: true
Query 18b >> 		 Runtime: 1204 ms - Result correct: true
Query 18c >> 		 Runtime: 1214 ms - Result correct: true
Query 19a >> 		 Runtime: 1205 ms - Result correct: true
Query 19b >> 		 Runtime: 1204 ms - Result correct: true
Query 19c >> 		 Runtime: 1208 ms - Result correct: true
Query 19d >> 		 Runtime: 1290 ms - Result correct: true
Query 20a >> 		 Runtime: 1218 ms - Result correct: true
Query 20b >> 		 Runtime: 1215 ms - Result correct: true
Query 20c >> 		 Runtime: 1226 ms - Result correct: true
Query 21a >> 		 Runtime: 1207 ms - Result correct: true
Query 21b >> 		 Runtime: 1207 ms - Result correct: true
Query 21c >> 		 Runtime: 1208 ms - Result correct: true
Query 22a >> 		 Runtime: 1212 ms - Result correct: true
Query 22b >> 		 Runtime: 1211 ms - Result correct: true
Query 22c >> 		 Runtime: 1214 ms - Result correct: true
Query 22d >> 		 Runtime: 1215 ms - Result correct: true
Query 23a >> 		 Runtime: 1205 ms - Result correct: true
Query 23b >> 		 Runtime: 1206 ms - Result correct: true
Query 23c >> 		 Runtime: 1206 ms - Result correct: true
Query 24a >> 		 Runtime: 1209 ms - Result correct: true
Query 24b >> 		 Runtime: 1209 ms - Result correct: true
Query 25a >> 		 Runtime: 1210 ms - Result correct: true
Query 25b >> 		 Runtime: 1207 ms - Result correct: true
Query 25c >> 		 Runtime: 1212 ms - Result correct: true
Query 26a >> 		 Runtime: 1227 ms - Result correct: true
Query 26b >> 		 Runtime: 1224 ms - Result correct: true
Query 26c >> 		 Runtime: 1230 ms - Result correct: true
Query 27a >> 		 Runtime: 1207 ms - Result correct: true
Query 27b >> 		 Runtime: 1206 ms - Result correct: true
Query 27c >> 		 Runtime: 1208 ms - Result correct: true
Query 28a >> 		 Runtime: 1213 ms - Result correct: true
Query 28b >> 		 Runtime: 1206 ms - Result correct: true
Query 28c >> 		 Runtime: 1209 ms - Result correct: true
Query 29a >> 		 Runtime: 1210 ms - Result correct: true
Query 29b >> 		 Runtime: 1209 ms - Result correct: true
Query 29c >> 		 Runtime: 1221 ms - Result correct: true
Query 30a >> 		 Runtime: 1211 ms - Result correct: true
Query 30b >> 		 Runtime: 1207 ms - Result correct: true
Query 30c >> 		 Runtime: 1234 ms - Result correct: true
Query 31a >> 		 Runtime: 1211 ms - Result correct: true
Query 31b >> 		 Runtime: 1208 ms - Result correct: true
Query 31c >> 		 Runtime: 1212 ms - Result correct: true
Query 32a >> 		 Runtime: 1203 ms - Result correct: true
Query 32b >> 		 Runtime: 1207 ms - Result correct: true
Query 33a >> 		 Runtime: 1209 ms - Result correct: true
Query 33b >> 		 Runtime: 1209 ms - Result correct: true
Query 33c >> 		 Runtime: 1217 ms - Result correct: true
"""

y_ARM = """
Query 1a >> 		 Runtime: 1206 ms - Result correct: true
Query 1b >> 		 Runtime: 1204 ms - Result correct: true
Query 1c >> 		 Runtime: 1206 ms - Result correct: true
Query 1d >> 		 Runtime: 1205 ms - Result correct: true
Query 2a >> 		 Runtime: 1209 ms - Result correct: true
Query 2b >> 		 Runtime: 1207 ms - Result correct: true
Query 2c >> 		 Runtime: 1206 ms - Result correct: true
Query 2d >> 		 Runtime: 1214 ms - Result correct: true
Query 3a >> 		 Runtime: 1205 ms - Result correct: true
Query 3b >> 		 Runtime: 1206 ms - Result correct: true
Query 3c >> 		 Runtime: 1206 ms - Result correct: true
Query 4a >> 		 Runtime: 1207 ms - Result correct: true
Query 4b >> 		 Runtime: 1206 ms - Result correct: true
Query 4c >> 		 Runtime: 1208 ms - Result correct: true
Query 5a >> 		 Runtime: 1202 ms - Result correct: true
Query 5b >> 		 Runtime: 1202 ms - Result correct: true
Query 5c >> 		 Runtime: 1204 ms - Result correct: true
Query 6a >> 		 Runtime: 1213 ms - Result correct: true
Query 6b >> 		 Runtime: 1212 ms - Result correct: true
Query 6c >> 		 Runtime: 1212 ms - Result correct: true
Query 6d >> 		 Runtime: 1212 ms - Result correct: true
Query 6e >> 		 Runtime: 1210 ms - Result correct: true
Query 6f >> 		 Runtime: 1239 ms - Result correct: true
Query 7a >> 		 Runtime: 1225 ms - Result correct: true
Query 7b >> 		 Runtime: 1214 ms - Result correct: true
Query 7c >> 		 Runtime: 1259 ms - Result correct: true
Query 8a >> 		 Runtime: 1207 ms - Result correct: true
Query 8b >> 		 Runtime: 1204 ms - Result correct: true
Query 8c >> 		 Runtime: 1243 ms - Result correct: true
Query 8d >> 		 Runtime: 1235 ms - Result correct: true
Query 9a >> 		 Runtime: 1206 ms - Result correct: true
Query 9b >> 		 Runtime: 1206 ms - Result correct: true
Query 9c >> 		 Runtime: 1233 ms - Result correct: true
Query 9d >> 		 Runtime: 1254 ms - Result correct: true
Query 10a >> 		 Runtime: 1208 ms - Result correct: true
Query 10b >> 		 Runtime: 1233 ms - Result correct: true
Query 10c >> 		 Runtime: 1245 ms - Result correct: true
Query 11a >> 		 Runtime: 1210 ms - Result correct: true
Query 11b >> 		 Runtime: 1206 ms - Result correct: true
Query 11c >> 		 Runtime: 1208 ms - Result correct: true
Query 11d >> 		 Runtime: 1209 ms - Result correct: true
Query 12a >> 		 Runtime: 1206 ms - Result correct: true
Query 12b >> 		 Runtime: 1211 ms - Result correct: true
Query 12c >> 		 Runtime: 1209 ms - Result correct: true
Query 13a >> 		 Runtime: 1223 ms - Result correct: true
Query 13b >> 		 Runtime: 1211 ms - Result correct: true
Query 13c >> 		 Runtime: 1210 ms - Result correct: true
Query 13d >> 		 Runtime: 1259 ms - Result correct: true
Query 14a >> 		 Runtime: 1209 ms - Result correct: true
Query 14b >> 		 Runtime: 1208 ms - Result correct: true
Query 14c >> 		 Runtime: 1209 ms - Result correct: true
Query 15a >> 		 Runtime: 1206 ms - Result correct: true
Query 15b >> 		 Runtime: 1214 ms - Result correct: true
Query 15c >> 		 Runtime: 1208 ms - Result correct: true
Query 15d >> 		 Runtime: 1209 ms - Result correct: true
Query 16a >> 		 Runtime: 1217 ms - Result correct: true
Query 16b >> 		 Runtime: 1261 ms - Result correct: true
Query 16c >> 		 Runtime: 1217 ms - Result correct: true
Query 16d >> 		 Runtime: 1215 ms - Result correct: true
Query 17a >> 		 Runtime: 1225 ms - Result correct: true
Query 17b >> 		 Runtime: 1216 ms - Result correct: true
Query 17c >> 		 Runtime: 1212 ms - Result correct: true
Query 17d >> 		 Runtime: 1213 ms - Result correct: true
Query 17e >> 		 Runtime: 1239 ms - Result correct: true
Query 17f >> 		 Runtime: 1238 ms - Result correct: true
Query 18a >> 		 Runtime: 1210 ms - Result correct: true
Query 18b >> 		 Runtime: 1206 ms - Result correct: true
Query 18c >> 		 Runtime: 1214 ms - Result correct: true
Query 19a >> 		 Runtime: 1207 ms - Result correct: true
Query 19b >> 		 Runtime: 1206 ms - Result correct: true
Query 19c >> 		 Runtime: 1208 ms - Result correct: true
Query 19d >> 		 Runtime: 1263 ms - Result correct: true
Query 20a >> 		 Runtime: 1215 ms - Result correct: true
Query 20b >> 		 Runtime: 1211 ms - Result correct: true
Query 20c >> 		 Runtime: 1228 ms - Result correct: true
Query 21a >> 		 Runtime: 1208 ms - Result correct: true
Query 21b >> 		 Runtime: 1208 ms - Result correct: true
Query 21c >> 		 Runtime: 1208 ms - Result correct: true
Query 22a >> 		 Runtime: 1211 ms - Result correct: true
Query 22b >> 		 Runtime: 1210 ms - Result correct: true
Query 22c >> 		 Runtime: 1213 ms - Result correct: true
Query 22d >> 		 Runtime: 1213 ms - Result correct: true
Query 23a >> 		 Runtime: 1207 ms - Result correct: true
Query 23b >> 		 Runtime: 1208 ms - Result correct: true
Query 23c >> 		 Runtime: 1207 ms - Result correct: true
Query 24a >> 		 Runtime: 1209 ms - Result correct: true
Query 24b >> 		 Runtime: 1210 ms - Result correct: true
Query 25a >> 		 Runtime: 1210 ms - Result correct: true
Query 25b >> 		 Runtime: 1209 ms - Result correct: true
Query 25c >> 		 Runtime: 1212 ms - Result correct: true
Query 26a >> 		 Runtime: 1219 ms - Result correct: true
Query 26b >> 		 Runtime: 1218 ms - Result correct: true
Query 26c >> 		 Runtime: 1220 ms - Result correct: true
Query 27a >> 		 Runtime: 1207 ms - Result correct: true
Query 27b >> 		 Runtime: 1207 ms - Result correct: true
Query 27c >> 		 Runtime: 1208 ms - Result correct: true
Query 28a >> 		 Runtime: 1211 ms - Result correct: true
Query 28b >> 		 Runtime: 1208 ms - Result correct: true
Query 28c >> 		 Runtime: 1209 ms - Result correct: true
Query 29a >> 		 Runtime: 1210 ms - Result correct: true
Query 29b >> 		 Runtime: 1212 ms - Result correct: true
Query 29c >> 		 Runtime: 1215 ms - Result correct: true
Query 30a >> 		 Runtime: 1211 ms - Result correct: true
Query 30b >> 		 Runtime: 1208 ms - Result correct: true
Query 30c >> 		 Runtime: 1213 ms - Result correct: true
Query 31a >> 		 Runtime: 1210 ms - Result correct: true
Query 31b >> 		 Runtime: 1209 ms - Result correct: true
Query 31c >> 		 Runtime: 1210 ms - Result correct: true
Query 32a >> 		 Runtime: 1204 ms - Result correct: true
Query 32b >> 		 Runtime: 1207 ms - Result correct: true
Query 33a >> 		 Runtime: 1211 ms - Result correct: true
Query 33b >> 		 Runtime: 1211 ms - Result correct: true
Query 33c >> 		 Runtime: 1213 ms - Result correct: true
"""

y_IBM = """
Query 1a >> 		 Runtime: 1203 ms - Result correct: true
Query 1b >> 		 Runtime: 1204 ms - Result correct: true
Query 1c >> 		 Runtime: 1204 ms - Result correct: true
Query 1d >> 		 Runtime: 1204 ms - Result correct: true
Query 2a >> 		 Runtime: 1209 ms - Result correct: true
Query 2b >> 		 Runtime: 1210 ms - Result correct: true
Query 2c >> 		 Runtime: 1206 ms - Result correct: true
Query 2d >> 		 Runtime: 1215 ms - Result correct: true
Query 3a >> 		 Runtime: 1213 ms - Result correct: true
Query 3b >> 		 Runtime: 1205 ms - Result correct: true
Query 3c >> 		 Runtime: 1207 ms - Result correct: true
Query 4a >> 		 Runtime: 1209 ms - Result correct: true
Query 4b >> 		 Runtime: 1205 ms - Result correct: true
Query 4c >> 		 Runtime: 1211 ms - Result correct: true
Query 5a >> 		 Runtime: 1200 ms - Result correct: true
Query 5b >> 		 Runtime: 1200 ms - Result correct: true
Query 5c >> 		 Runtime: 1203 ms - Result correct: true
Query 6a >> 		 Runtime: 1225 ms - Result correct: true
Query 6b >> 		 Runtime: 1224 ms - Result correct: true
Query 6c >> 		 Runtime: 1223 ms - Result correct: true
Query 6d >> 		 Runtime: 1225 ms - Result correct: true
Query 6e >> 		 Runtime: 1239 ms - Result correct: true
Query 6f >> 		 Runtime: 1262 ms - Result correct: true
Query 7a >> 		 Runtime: 1250 ms - Result correct: true
Query 7b >> 		 Runtime: 1280 ms - Result correct: true
Query 7c >> 		 Runtime: 1310 ms - Result correct: true
Query 8a >> 		 Runtime: 1207 ms - Result correct: true
Query 8b >> 		 Runtime: 1202 ms - Result correct: true
Query 8c >> 		 Runtime: 1314 ms - Result correct: true
Query 8d >> 		 Runtime: 1359 ms - Result correct: true
Query 9a >> 		 Runtime: 1207 ms - Result correct: true
Query 9b >> 		 Runtime: 1205 ms - Result correct: true
Query 9c >> 		 Runtime: 1391 ms - Result correct: true
Query 9d >> 		 Runtime: 1482 ms - Result correct: true
Query 10a >> 		 Runtime: 1207 ms - Result correct: true
Query 10b >> 		 Runtime: 1317 ms - Result correct: true
Query 10c >> 		 Runtime: 1335 ms - Result correct: true
Query 11a >> 		 Runtime: 1208 ms - Result correct: true
Query 11b >> 		 Runtime: 1205 ms - Result correct: true
Query 11c >> 		 Runtime: 1209 ms - Result correct: true
Query 11d >> 		 Runtime: 1211 ms - Result correct: true
Query 12a >> 		 Runtime: 1205 ms - Result correct: true
Query 12b >> 		 Runtime: 1296 ms - Result correct: true
Query 12c >> 		 Runtime: 1210 ms - Result correct: true
Query 13a >> 		 Runtime: 1236 ms - Result correct: true
Query 13b >> 		 Runtime: 1242 ms - Result correct: true
Query 13c >> 		 Runtime: 1237 ms - Result correct: true
Query 13d >> 		 Runtime: 1542 ms - Result correct: true
Query 14a >> 		 Runtime: 1210 ms - Result correct: true
Query 14b >> 		 Runtime: 1206 ms - Result correct: true
Query 14c >> 		 Runtime: 1210 ms - Result correct: true
Query 15a >> 		 Runtime: 1206 ms - Result correct: true
Query 15b >> 		 Runtime: 1206 ms - Result correct: true
Query 15c >> 		 Runtime: 1208 ms - Result correct: true
Query 15d >> 		 Runtime: 1212 ms - Result correct: true
Query 16a >> 		 Runtime: 1237 ms - Result correct: true
Query 16b >> 		 Runtime: 1647 ms - Result correct: true
Query 16c >> 		 Runtime: 1240 ms - Result correct: true
Query 16d >> 		 Runtime: 1237 ms - Result correct: true
Query 17a >> 		 Runtime: 1251 ms - Result correct: true
Query 17b >> 		 Runtime: 1233 ms - Result correct: true
Query 17c >> 		 Runtime: 1236 ms - Result correct: true
Query 17d >> 		 Runtime: 1349 ms - Result correct: true
Query 17e >> 		 Runtime: 1464 ms - Result correct: true
Query 17f >> 		 Runtime: 1319 ms - Result correct: true
Query 18a >> 		 Runtime: 1223 ms - Result correct: true
Query 18b >> 		 Runtime: 1205 ms - Result correct: true
Query 18c >> 		 Runtime: 1217 ms - Result correct: true
Query 19a >> 		 Runtime: 1207 ms - Result correct: true
Query 19b >> 		 Runtime: 1206 ms - Result correct: true
Query 19c >> 		 Runtime: 1214 ms - Result correct: true
Query 19d >> 		 Runtime: 1575 ms - Result correct: true
Query 20a >> 		 Runtime: 1230 ms - Result correct: true
Query 20b >> 		 Runtime: 1229 ms - Result correct: true
Query 20c >> 		 Runtime: 1240 ms - Result correct: true
Query 21a >> 		 Runtime: 1208 ms - Result correct: true
Query 21b >> 		 Runtime: 1208 ms - Result correct: true
Query 21c >> 		 Runtime: 1210 ms - Result correct: true
Query 22a >> 		 Runtime: 1213 ms - Result correct: true
Query 22b >> 		 Runtime: 1213 ms - Result correct: true
Query 22c >> 		 Runtime: 1223 ms - Result correct: true
Query 22d >> 		 Runtime: 1316 ms - Result correct: true
Query 23a >> 		 Runtime: 1208 ms - Result correct: true
Query 23b >> 		 Runtime: 1208 ms - Result correct: true
Query 23c >> 		 Runtime: 1208 ms - Result correct: true
Query 24a >> 		 Runtime: 1215 ms - Result correct: true
Query 24b >> 		 Runtime: 1325 ms - Result correct: true
Query 25a >> 		 Runtime: 1222 ms - Result correct: true
Query 25b >> 		 Runtime: 1211 ms - Result correct: true
Query 25c >> 		 Runtime: 1229 ms - Result correct: true
Query 26a >> 		 Runtime: 1241 ms - Result correct: true
Query 26b >> 		 Runtime: 1263 ms - Result correct: true
Query 26c >> 		 Runtime: 1246 ms - Result correct: true
Query 27a >> 		 Runtime: 1212 ms - Result correct: true
Query 27b >> 		 Runtime: 1208 ms - Result correct: true
Query 27c >> 		 Runtime: 1209 ms - Result correct: true
Query 28a >> 		 Runtime: 1214 ms - Result correct: true
Query 28b >> 		 Runtime: 1207 ms - Result correct: true
Query 28c >> 		 Runtime: 1210 ms - Result correct: true
Query 29a >> 		 Runtime: 1265 ms - Result correct: true
Query 29b >> 		 Runtime: 1215 ms - Result correct: true
Query 29c >> 		 Runtime: 1318 ms - Result correct: true
Query 30a >> 		 Runtime: 1251 ms - Result correct: true
Query 30b >> 		 Runtime: 1209 ms - Result correct: true
Query 30c >> 		 Runtime: 1258 ms - Result correct: true
Query 31a >> 		 Runtime: 1220 ms - Result correct: true
Query 31b >> 		 Runtime: 1212 ms - Result correct: true
Query 31c >> 		 Runtime: 1224 ms - Result correct: true
Query 32a >> 		 Runtime: 1204 ms - Result correct: true
Query 32b >> 		 Runtime: 1208 ms - Result correct: true
Query 33a >> 		 Runtime: 1215 ms - Result correct: true
Query 33b >> 		 Runtime: 1216 ms - Result correct: true
Query 33c >> 		 Runtime: 1223 ms - Result correct: true
"""

y_INTEL = """
Query 1a >> 		 Runtime: 1205 ms - Result correct: true
Query 1b >> 		 Runtime: 1204 ms - Result correct: true
Query 1c >> 		 Runtime: 1205 ms - Result correct: true
Query 1d >> 		 Runtime: 1204 ms - Result correct: true
Query 2a >> 		 Runtime: 1212 ms - Result correct: true
Query 2b >> 		 Runtime: 1211 ms - Result correct: true
Query 2c >> 		 Runtime: 1208 ms - Result correct: true
Query 2d >> 		 Runtime: 1227 ms - Result correct: true
Query 3a >> 		 Runtime: 1207 ms - Result correct: true
Query 3b >> 		 Runtime: 1206 ms - Result correct: true
Query 3c >> 		 Runtime: 1208 ms - Result correct: true
Query 4a >> 		 Runtime: 1210 ms - Result correct: true
Query 4b >> 		 Runtime: 1206 ms - Result correct: true
Query 4c >> 		 Runtime: 1212 ms - Result correct: true
Query 5a >> 		 Runtime: 1202 ms - Result correct: true
Query 5b >> 		 Runtime: 1201 ms - Result correct: true
Query 5c >> 		 Runtime: 1204 ms - Result correct: true
Query 6a >> 		 Runtime: 1222 ms - Result correct: true
Query 6b >> 		 Runtime: 1226 ms - Result correct: true
Query 6c >> 		 Runtime: 1225 ms - Result correct: true
Query 6d >> 		 Runtime: 1227 ms - Result correct: true
Query 6e >> 		 Runtime: 1226 ms - Result correct: true
Query 6f >> 		 Runtime: 1279 ms - Result correct: true
Query 7a >> 		 Runtime: 1271 ms - Result correct: true
Query 7b >> 		 Runtime: 1235 ms - Result correct: true
Query 7c >> 		 Runtime: 1416 ms - Result correct: true
Query 8a >> 		 Runtime: 1207 ms - Result correct: true
Query 8b >> 		 Runtime: 1203 ms - Result correct: true
Query 8c >> 		 Runtime: 1339 ms - Result correct: true
Query 8d >> 		 Runtime: 1296 ms - Result correct: true
Query 9a >> 		 Runtime: 1207 ms - Result correct: true
Query 9b >> 		 Runtime: 1206 ms - Result correct: true
Query 9c >> 		 Runtime: 1250 ms - Result correct: true
Query 9d >> 		 Runtime: 1304 ms - Result correct: true
Query 10a >> 		 Runtime: 1208 ms - Result correct: true
Query 10b >> 		 Runtime: 1245 ms - Result correct: true
Query 10c >> 		 Runtime: 1347 ms - Result correct: true
Query 11a >> 		 Runtime: 1211 ms - Result correct: true
Query 11b >> 		 Runtime: 1207 ms - Result correct: true
Query 11c >> 		 Runtime: 1211 ms - Result correct: true
Query 11d >> 		 Runtime: 1213 ms - Result correct: true
Query 12a >> 		 Runtime: 1206 ms - Result correct: true
Query 12b >> 		 Runtime: 1216 ms - Result correct: true
Query 12c >> 		 Runtime: 1211 ms - Result correct: true
Query 13a >> 		 Runtime: 1249 ms - Result correct: true
Query 13b >> 		 Runtime: 1222 ms - Result correct: true
Query 13c >> 		 Runtime: 1222 ms - Result correct: true
Query 13d >> 		 Runtime: 1376 ms - Result correct: true
Query 14a >> 		 Runtime: 1213 ms - Result correct: true
Query 14b >> 		 Runtime: 1208 ms - Result correct: true
Query 14c >> 		 Runtime: 1213 ms - Result correct: true
Query 15a >> 		 Runtime: 1208 ms - Result correct: true
Query 15b >> 		 Runtime: 1207 ms - Result correct: true
Query 15c >> 		 Runtime: 1210 ms - Result correct: true
Query 15d >> 		 Runtime: 1215 ms - Result correct: true
Query 16a >> 		 Runtime: 1249 ms - Result correct: true
Query 16b >> 		 Runtime: 1362 ms - Result correct: true
Query 16c >> 		 Runtime: 1246 ms - Result correct: true
Query 16d >> 		 Runtime: 1243 ms - Result correct: true
Query 17a >> 		 Runtime: 1255 ms - Result correct: true
Query 17b >> 		 Runtime: 1239 ms - Result correct: true
Query 17c >> 		 Runtime: 1230 ms - Result correct: true
Query 17d >> 		 Runtime: 1231 ms - Result correct: true
Query 17e >> 		 Runtime: 1300 ms - Result correct: true
Query 17f >> 		 Runtime: 1320 ms - Result correct: true
Query 18a >> 		 Runtime: 1220 ms - Result correct: true
Query 18b >> 		 Runtime: 1206 ms - Result correct: true
Query 18c >> 		 Runtime: 1223 ms - Result correct: true
Query 19a >> 		 Runtime: 1208 ms - Result correct: true
Query 19b >> 		 Runtime: 1207 ms - Result correct: true
Query 19c >> 		 Runtime: 1213 ms - Result correct: true
Query 19d >> 		 Runtime: 1343 ms - Result correct: true
Query 20a >> 		 Runtime: 1233 ms - Result correct: true
Query 20b >> 		 Runtime: 1229 ms - Result correct: true
Query 20c >> 		 Runtime: 1245 ms - Result correct: true
Query 21a >> 		 Runtime: 1212 ms - Result correct: true
Query 21b >> 		 Runtime: 1212 ms - Result correct: true
Query 21c >> 		 Runtime: 1214 ms - Result correct: true
Query 22a >> 		 Runtime: 1219 ms - Result correct: true
Query 22b >> 		 Runtime: 1217 ms - Result correct: true
Query 22c >> 		 Runtime: 1222 ms - Result correct: true
Query 22d >> 		 Runtime: 1223 ms - Result correct: true
Query 23a >> 		 Runtime: 1210 ms - Result correct: true
Query 23b >> 		 Runtime: 1209 ms - Result correct: true
Query 23c >> 		 Runtime: 1209 ms - Result correct: true
Query 24a >> 		 Runtime: 1215 ms - Result correct: true
Query 24b >> 		 Runtime: 1211 ms - Result correct: true
Query 25a >> 		 Runtime: 1218 ms - Result correct: true
Query 25b >> 		 Runtime: 1213 ms - Result correct: true
Query 25c >> 		 Runtime: 1222 ms - Result correct: true
Query 26a >> 		 Runtime: 1255 ms - Result correct: true
Query 26b >> 		 Runtime: 1247 ms - Result correct: true
Query 26c >> 		 Runtime: 1262 ms - Result correct: true
Query 27a >> 		 Runtime: 1211 ms - Result correct: true
Query 27b >> 		 Runtime: 1209 ms - Result correct: true
Query 27c >> 		 Runtime: 1213 ms - Result correct: true
Query 28a >> 		 Runtime: 1219 ms - Result correct: true
Query 28b >> 		 Runtime: 1210 ms - Result correct: true
Query 28c >> 		 Runtime: 1213 ms - Result correct: true
Query 29a >> 		 Runtime: 1216 ms - Result correct: true
Query 29b >> 		 Runtime: 1216 ms - Result correct: true
Query 29c >> 		 Runtime: 1239 ms - Result correct: true
Query 30a >> 		 Runtime: 1216 ms - Result correct: true
Query 30b >> 		 Runtime: 1214 ms - Result correct: true
Query 30c >> 		 Runtime: 1219 ms - Result correct: true
Query 31a >> 		 Runtime: 1219 ms - Result correct: true
Query 31b >> 		 Runtime: 1213 ms - Result correct: true
Query 31c >> 		 Runtime: 1221 ms - Result correct: true
Query 32a >> 		 Runtime: 1206 ms - Result correct: true
Query 32b >> 		 Runtime: 1210 ms - Result correct: true
Query 33a >> 		 Runtime: 1214 ms - Result correct: true
Query 33b >> 		 Runtime: 1217 ms - Result correct: true
Query 33c >> 		 Runtime: 1226 ms - Result correct: true
"""

def draw_linear_regression(x, y, title):
    ls_x = []
    ls_y = []
    for line in x.split('\n'):
        if p := re.search(r'Query \d+\w[^\d+]+(\d+) ms.+', line):
            ls_x.append(int(p.group(1)))
    for line in y.split('\n'):
        if q := re.search(r'Query \d+\w[^\d+]+(\d+) ms.+', line):
            ls_y.append(int(q.group(1)) - 1200)
    print(ls_x)
    print(ls_y)

    slope, intercept, r_value, p_value, std_err = linregress(ls_x, ls_y)
    print(f"斜率: {slope}, 截距: {intercept}, R²: {r_value**2}")

    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    x_fit = np.linspace(min(ls_x), max(ls_x), 100)  # 生成平滑的 x 值
    y_fit = slope * x_fit + intercept         # 计算回归直线的 y 值
    # 绘制散点图
    plt.scatter(ls_x, ls_y, color='blue', label='数据点')
    # 绘制回归直线
    plt.plot(x_fit, y_fit, color='red', linestyle='-', label=f'回归线: y={slope:.2f}x+{intercept:.2f}')
    # 显示 R² 值
    plt.text(min(ls_x), max(ls_y) * 0.9, f'R² = {r_value**2:.3f}', fontsize=12, color='black')
    # 添加标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # 显示图像
    plt.show()


if __name__ == '__main__':
    draw_linear_regression(x, y_AMD, 'AMD')
    draw_linear_regression(x, y_ARM, 'ARM')
    draw_linear_regression(x, y_IBM, 'IBM')
    draw_linear_regression(x, y_INTEL, 'INTEL')
