from math import floor
import time


def ft_progress(lst):
    print("\x1b[?25l") # hide cursor
    i = 1
    start = time.time()
    while i <= len(lst):
        yield i
        pourc = int(i * 100 / len(lst))
        nb = int(i * 20 / len(lst))
        arrow = ">".rjust(nb, "=")
        top = time.time() - start
        eta = (len(lst) * top / i) - top
        if eta <= 100:
            etah = 0
            etam = 0
            etas = eta
        elif eta > 100 and eta < 3600:
            etah = 0
            etam = floor(eta / 60)
            etas = eta - (etam * 60)
        else:
            etah = floor(eta / (60 * 60))
            etam = floor((eta - (etah * 60 * 60)) / 60)
            etas = eta - (etah * 60 * 60) - (etam * 60)
        if top <= 100:
            toph = 0
            topm = 0
            tops = top
        elif top > 100 and top < 3600:
            toph = 0
            topm = floor(top / 60)
            tops = top - (topm * 60)
        else:
            toph = floor(top / (60 * 60))
            topm = floor((top - (toph * 60 * 60)) / 60)
            tops = top - (toph * 60 * 60) - (topm * 60)
        label = f"ETA:"
        if etah > 0:
            label = f"{label} {etah}h"
        if etam > 0 or etah > 0:
            label = f"{label} {etam:02}mn"
        label = f"{label} {etas:05.2f}s [{pourc:3}%] [{arrow:<20}] {i}/{len(lst)} | elapsed time"
        if toph > 0:
            label = f"{label} {toph}h"
        if topm > 0 or toph > 0:
            label = f"{label} {topm}mn"
        label = f"{label} {tops:05.2f}s    "
        print(f"{label}", end='\r', flush=True)
        i += 1
    print("\x1b[?25h") #show cursor
