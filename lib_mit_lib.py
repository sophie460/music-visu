import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle
import pyaudio
import pygame
import math
from collections import deque

recent_amps = deque(maxlen=30)  

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
    signal = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    if not np.isfinite(signal).all():
        print("Signal enthält nicht-endliche Werte")

    return signal

def extract_features(signal, rate):
    max_signal = np.max(np.abs(signal))
    if max_signal == 0:
        max_signal = 1 #damit nicht durch 0 geteilt wird
    signal = signal / max_signal

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(signal).all():
        return None, None, None, None, None

    n_fft = 256
    if len(signal) < n_fft:
        return None, None, None, None, None
    hop_length = 128
           
    try:
        stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
        mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(S=stft, sr=rate)
        mel = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=n_fft, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=rate, n_fft=n_fft, hop_length=hop_length)
        try: 
            harmonic = librosa.effects.harmonic(signal)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=rate)
        except Exception as e:
            print("Tonnetz extraction failed:", e)
            tonnetz = None
    except ValueError as e:
        print("drecks feature extraction buggt {e}")
        return None, None, None, None, None
    
        
    return mfccs, chroma, mel, contrast, tonnetz

def detect_music_elements(mfccs, chroma, mel, contrast, tonnetz):
    if mfccs is None or chroma is None or mel is None or contrast is None or tonnetz is None:
        return 0, 0
    
    # Simple heuristic-based detection (for illustration purposes)
    bass = np.mean(mel[:10])  # Low frequencies
    drums = np.mean(contrast)  # High spectral contrast

    bass = (bass ** 0.8) * 0.8 #die beiden optio
    drums = (drums ** 0.8) * 0.5
    return bass, drums

def draw(amplitude, bass, drums):
    screen.fill((0, 0, 0))

    points = []
    for x in range(screen_width):
        y = screen_height/2 + int(amplitude * math.sin(x*0.02))
        points.append((x, y))
    pygame.draw.lines(screen, (255, 255, 255), False, points, 2)

    bass_line = []
    for x in range(screen_width):
        y = screen_height/2 + int(bass * math.sin(x*0.02 + math.pi/2))
        bass_line.append((x, y))
    pygame.draw.lines(screen, (255, 0, 0), False, bass_line, 2)

    drum_line = []
    for x in range(screen_width):
        y = screen_height/2 + int(drums * math.sin(x*0.02 + math.pi))
        drum_line.append((x, y))
    pygame.draw.lines(screen, (0, 0, 255), False, drum_line, 2)

    pygame.display.flip()

running = True
amplitude = 100

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    signal = mic_input()
    raw_amp = np.mean(np.abs(signal))
    recent_amps.append(raw_amp) 
    max_amp = max(recent_amps) if recent_amps else 1e-6
    normalized_amp = raw_amp / max_amp

    amplitude = max(2, (normalized_amp ** 1.2) * 100)  # Adjusted amplitude calculation

    # amplitude_adjust = (raw_amp ** 1.2) * 100  # Exponential boost, hat ganz gut funktioniert
    # amplitude = max(2, amplitude_adjust)

    # amplitude_adjust = np.mean(np.abs(signal)) / 10 #30 less sensible
    # amplitude = max(2, amplitude_adjust)

    features = extract_features(signal, rate)
    if any(f is None for f in features):
        continue
    
    mfccs, chroma, mel, contrast, tonnetz = features
    bass, drums = detect_music_elements(mfccs, chroma, mel, contrast, tonnetz)

    draw(amplitude, bass, drums)
    clock.tick(60)

#pygame.quit()
