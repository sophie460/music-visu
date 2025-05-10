import pandas as pd
import numpy as np
import matplotlib.pylab as plt
# import seaborn as sns
# from glob import glob
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle
import pyaudio
import pygame
import math

# pygame setup
screen_width = 1200
screen_height = 600
pygame.init()
pygame.display.set_caption("pygame window")
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# pyaudio setup
chunk = 1024
format = pyaudio.paInt16
channels = 1
rate = 44100
p = pyaudio.PyAudio()
stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

def mic_input():
    data = stream.read(chunk)
    rms = 0 #root mean squared
    for i in range(0, len(data), 2):
        sample = int.from_bytes(data[i:i +2], byteorder="little", signed=True)
        rms += sample * sample
    rms = math.sqrt(rms / (chunk / 2))
    return rms

def draw(amplitude):
    screen.fill((0, 0, 0))
    points = []
    if amplitude > 10:
        for x in range(screen_width):
            y = screen_height/2 + int(amplitude * math.sin(x*0.02))
            points.append((x, y))
    else:
        points.append((0, screen_height/2))
        points.append((screen_width, screen_height/2))

    pygame.draw.lines(screen, (255, 255, 255), False, points, 2)
    pygame.display.flip()

running = True
amplitude = 100

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    amplitude_adjust = mic_input() / 30
    amplitude = max(2, amplitude_adjust)

    draw(amplitude)
    print(mic_input())
    clock.tick(60)
pygame.quit()

"""
#bisschen farben stuff
# sns.set_theme(style="white", palette=None)
# color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# audio_file = glob("/music visualizer/Bru-C  - You & I [Music Video].wav") glob immer liste; index mitangeben
test = "Bru-C  - You & I [Music Video].wav"

y, sr = librosa.load(test)
# print(f"y: {y[:10]}")
# print(f"shape y: {y.shape}")
# print(f"sr: {sr}")

pd.Series(y).plot(figsize=(10, 5), lw=1, title="spectogram von y")
# plt.show()
"""
