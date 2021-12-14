"""
Copyright © 2021 Wallbreaker5th

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nf
from manimlib import *

N=4

svg=SVGMobject(file_name="tri.svg")[0]
x=np.array([svg.point_from_proportion(i/N)[0] for i in range(N)])
y=np.array([svg.point_from_proportion(i/N)[1] for i in range(N)])
a=np.array([i+1j*j for i,j in zip(x,y)])
a=a/np.array([np.exp(i*(-N//2)*np.pi*2j/N) for i in range(N)])

fa=nf.fft(a)/N

p=np.array([sum(fa[i]*np.exp((i-N//2)*j*np.pi*2j/(N*6)) for i in range(N)) for j in range(N*6)])

plt.plot(x,y,'o:')
plt.plot([i.real for i in p],[i.imag for i in p])
plt.show()
