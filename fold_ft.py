"""
Copyright © 2021 Wallbreaker5th

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Tuple
from manimlib import *
import numpy as np
import time
import numpy.fft as nf

timestamp = 0
DEFAULT_STROKE_WIDTH = 2


def get_rand():
    return 1+np.random.random()


def calc(p: np.ndarray, q: np.ndarray, r: float) -> Tuple[np.ndarray, np.ndarray]:
    time0 = time.time()
    dis = np.linalg.norm(p-q)
    if r < (dis/2)*(1-1e-8):
        # assert(0)
        pass
    h = np.sqrt(max(r**2-(dis/2)**2, 0))
    v = (q-p)/dis*h
    v = np.array([v[1], -v[0], 0])
    mid = (p+q)/2
    p1, p2 = mid+v, mid-v
    return (p1, p2)


def reflect(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    l = np.linalg.norm(a-b)-np.dot(p-a, b-a)/np.linalg.norm(b-a)*2
    return p+l*(b-a)/np.linalg.norm(b-a)


def set_line(mob, p, q):
    mob.data["points"][0] = p
    mob.data["points"][1] = q
    mob.data["points"][2] = q
    # mob.data["points"]=np.array([p, midpoint(p, q), q])


class Fold(VMobject):
    opacity = -1
    color = -1
    stroke_with = -1

    def set_color(self, color, recurse=False, escape=None):
        global timestamp
        self.color = timestamp
        for i in self.submobjects:
            if i.color != timestamp:
                if (recurse or not isinstance(i, Fold)) and (not escape or not i in escape):
                    if isinstance(i, Fold):
                        i.set_color(color, recurse, escape)
                    else:
                        i.set_color(color, recurse)
        return self

    def set_opacity(self, opacity, recurse=False, escape=None):
        global timestamp
        self.opacity = timestamp
        self.opacity_val = opacity
        self.data["opacity"] = opacity
        for i in self.submobjects:
            if i.opacity != timestamp:
                if (recurse or not isinstance(i, Fold)) and (not escape or not i in escape):
                    if isinstance(i, Fold):
                        i.set_opacity(opacity, recurse, escape)
                    else:
                        i.set_opacity(opacity, recurse)
        return self

    def set_stroke_width(self, width, recurse=False, escape=None):
        global timestamp
        self.stroke_width = timestamp
        for i in self.submobjects:
            if i.stroke_width != timestamp:
                if (recurse or not isinstance(i, Fold)) and (not escape or not i in escape):
                    if isinstance(i, Fold):
                        i.set_stroke_width(width, recurse, escape)
                    else:
                        i.set_stroke(width=width)


class Funiop(Fold):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamp = -1

    def is_newest(self):
        global timestamp, qaq
        return self.timestamp == timestamp

    def get_output(self):
        return self.output

    def get_input(self):
        global timestamp
        if isinstance(self.pre, Fold) and not self.pre.is_newest():
            self.pre.fupdate()
            self.pre.timestamp = timestamp
        self.input = self.pre.get_output() if isinstance(self.pre, Fold) else self.pre()
        return self.input

    def set_self_color(self, color):
        self.set_color(color, recurse=True, escape={self.pre})
        return self

    def set_self_opacity(self, opacity):
        self.set_opacity(opacity, recurse=True, escape={self.pre})
        return self

    def set_self_stroke_width(self, stroke_width):
        self.set_stroke_width(stroke_width, recurse=True, escape={self.pre})
        return self


class Frotate(Funiop):
    def fupdate(self):
        t = self.get_input()*self.rate
        pX = np.array([0, 0, 0])
        r = self.r
        pO = np.array([r*np.cos(t), r*np.sin(t), 0])
        self.output = pO
        set_line(self.submobjects[1], pX, pO)

    def __init__(self, r, rate, pre, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.rate = rate
        self.pre = pre

        self.add(SmallDot(**kwargs))
        self.add(Line(**kwargs))

        self.fupdate()


class Frotate_2(Funiop):
    def fupdate(self):
        self.submobjects[0].fupdate()
        self.submobjects[1].fupdate()
        set_line(self.submobjects[2], self.submobjects[0].output,
                 self.submobjects[1].output)
        self.output = self.submobjects[1].output

    def __init__(self, v0, r, rate, pre, **kwargs):
        super().__init__(**kwargs)
        self.r = abs(v0)
        self.rate = rate
        self.pre = pre

        self.add(Frotate(r, rate, pre, **kwargs))
        self.add(Frotate(self.r, 1, lambda: pre()*rate +
                 np.angle(v0), **kwargs).set_color(RED))
        self.add(Line(**kwargs).set_color(YELLOW))

        self.fupdate()


class Fmulti(Funiop):
    def fupdate(self):
        t = self.get_input()
        pA = np.array([self.r1*np.cos(t*self.rate1),
                      self.r1*np.sin(t*self.rate1), 0])
        pB = np.array([self.r2*np.cos(t*self.rate2),
                      self.r2*np.sin(t*self.rate2), 0])
        pC = np.array([self.r3*np.cos(t*self.rate3),
                      self.r3*np.sin(t*self.rate3), 0])
        pO = np.array([0, 0, 0])

        pD = reflect(pO, pA, pB)
        pE = reflect(pO, pB, pC)
        set_line(self[0], pA, pD)
        set_line(self[1], pB, pD)
        set_line(self[2], pC, pE)

    def __init__(self, r1, r2, r3, rate1, rate2, rate3, pre, **kwargs):
        super().__init__(**kwargs)
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.rate1 = rate1
        self.rate2 = rate2
        self.rate3 = rate3
        self.pre = pre

        self.add(Line(**kwargs))
        self.add(Line(**kwargs))
        self.add(Line(**kwargs))

        self.fupdate()


class Fbinop(Fold):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamp = -1

    def is_newest(self):
        global timestamp, qaq
        return self.timestamp == timestamp

    def get_output(self):
        return self.output

    def get_input(self):
        global timestamp
        if isinstance(self.in1, Fold) and not self.in1.is_newest():
            self.in1.fupdate()
            self.in1.timestamp = timestamp
        if isinstance(self.in2, Fold) and not self.in2.is_newest():
            self.in2.fupdate()
            self.in2.timestamp = timestamp
        self.input = (self.in1.get_output() if isinstance(self.in1, Fold) else self.in1(),
                      self.in2.get_output() if isinstance(self.in2, Fold) else self.in2())
        return self.input

    def set_self_color(self, color):
        self.set_color(color, recurse=True, escape={self.in1, self.in2})
        return self

    def set_self_opacity(self, opacity):
        self.set_opacity(opacity, recurse=True, escape={self.in1, self.in2})
        return self

    def set_self_stroke_width(self, stroke_width):
        self.set_stroke_width(stroke_width, recurse=True,
                              escape={self.in1, self.in2})
        return self


def pluser_updater(self, pI1, pI2, r1=0, r2=0):
    pX = np.array([0, 0, 0])
    pO = np.array(pI1+pI2)
    pA = calc(pX, pI1, self.a)[r1]
    pB = calc(pX, pI2, self.a)[r2]
    pC = pA+pB
    pD = pB+pI1
    pE = pI2+pA
    set_line(self.submobjects[1], pX, pB)
    set_line(self.submobjects[2], pB, pI2)
    set_line(self.submobjects[3], pA, pC)
    set_line(self.submobjects[4], pC, pE)
    set_line(self.submobjects[5], pI1, pD)
    set_line(self.submobjects[6], pD, pO)
    set_line(self.submobjects[7], pX, pA)
    set_line(self.submobjects[8], pA, pI1)
    set_line(self.submobjects[9], pB, pC)
    set_line(self.submobjects[10], pC, pD)
    set_line(self.submobjects[11], pI2, pE)
    set_line(self.submobjects[12], pE, pO)


class Fplus(Fbinop):
    def fupdate(self):
        pX = np.array([0, 0, 0])
        I1, I2 = self.get_input()
        self.output = I1+I2
        pluser_updater(self, I1, I2, 0, 0)

    def __init__(self, a, in1, in2, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.in1 = in1
        self.in2 = in2

        self.add(SmallDot(**kwargs))  # X
        for i in range(12):
            self.add(Line(**kwargs))

        if isinstance(in1, Fold):
            self.add(in1)
            self.in1 = self.submobjects[-1]
        if isinstance(in2, Fold):
            self.add(in2)
            self.in2 = self.submobjects[-1]

        self.fupdate()


class Ffourier(Funiop):
    def fupdate(self):
        for i in self.submobjects:
            i.fupdate()
        self.output = self.res.get_output()

    def __init__(self, fft, r0, k, pre, **kwargs):
        super().__init__(**kwargs)
        self.fft = fft
        self.n = fft.shape[0]//2
        n = self.n
        for i in range(-n, n):
            self.add(Frotate_2(fft[i+n], r0*k**i, i, pre, **kwargs).set_color(
                interpolate_color(WHITE, WHITE, (i+n)/(n*2)), recurse=True
            ))
        for i in range(-n, n-2):
            self.add(Fmulti(r0*k**(i+0), r0*k**(i+1), r0*k **
                     (i+2), i+0, i+1, i+2, pre, **kwargs).set_color(
                interpolate_color(ORANGE, ORANGE, (i+n)/(n*2))
            ))
        s = abs(fft[0])
        for i in range(-n, n-1):
            s += abs(fft[i+n+1])
            self.add(Fplus(s+1, self[i+n if i == -n else -1], self[i+n+1], **kwargs).set_color(
                interpolate_color('#0080FF', '#8080FF', (i+n)%4/3)
            ).set_opacity(0.3))

        self.res = self.submobjects[-1]
        self.submobjects = self.submobjects[n*4-2:]+self.submobjects[:n*4-2]

        self.fupdate()


class Fscene(Scene):
    def construct(self):
        global timestamp
        time_total = 10
        speed = 1
        self.camera.frame.scale(0.5)

        N = 4
        svg = SVGMobject(file_name="tri.svg")[0].set_height(2).move_to(ORIGIN)
        x = np.array([svg.point_from_proportion(i/N)[0] for i in range(N)])
        y = np.array([svg.point_from_proportion(i/N)[1] for i in range(N)])
        a = np.array([i+1j*j for i, j in zip(x, y)])
        a = a/np.array([np.exp(i*(-N//2)*np.pi*2j/N) for i in range(N)])
        fa = nf.fft(a)/N

        fres = Ffourier(fa, 1, 0.8, lambda: timestamp, stroke_width=2)
        self.add(fres)
        print(sum(isinstance(i, Line) for i in fres.family))

        path = VMobject(stroke_width=4)
        path.set_points([fres.get_output()])

        pen = SmallDot(color='#0080ff', radius=0.02)
        self.add(pen)

        def path_upd(mob):
            return mob.add_line_to(fres.get_output())
        self.add(path)

        def pen_upd(mob):
            return mob.move_to(fres.get_output())
        self.add(pen)

        def time_updater(mob, dt):
            global timestamp
            timestamp += dt/time_total*TAU*speed
            mob.fupdate()

        fres.add_updater(time_updater)
        path.add_updater(path_upd)
        pen.add_updater(pen_upd)

        self.wait(time_total)

        path.remove_updater(path_upd)
        fres.remove_updater(time_updater)
        pen.remove_updater(pen_upd)

        self.wait(3)
