import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import pygame
import pretty_midi
from threading import Thread

class Player(Thread):
    def __init__(self, freq=44100, bitsize=-16, channels=2, buffer=1024, volume=0.8, fadeout=False):
        # mixer config
        self.freq = 44100  # audio CD quality
        self.bitsize = -16   # unsigned 16 bit
        self.channels = 2  # 1 is mono, 2 is stereo
        self.buffer = 1024   # number of samples
        self.volume = volume
        self.fadeout = fadeout
        self.thread = None

        # pygame.mixer.init(freq, bitsize, channels, buffer)
        # pygame.mixer.music.set_volume(self.volume)

    def play_song(self, filename):
        if filename:
            try:
                clock = pygame.time.Clock()
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    clock.tick(30) # check if playback has finished
            except KeyboardInterrupt:
                # if user hits Ctrl/C then exit
                # (works only in console mode)
                if self.fadeout:
                    pygame.mixer.music.fadeout(1000)
                pygame.mixer.music.stop()
                raise SystemExit

    def play(self, filename):
        self.thread = Thread(target=self.play_song, args=(filename, ))
        self.thread.start()
        
    def stop(self):
        pygame.mixer.music.stop()


if __name__ == '__main__':
    player = Player()
    player.play('/home/bryan/amici/POP909-Dataset/POP909/001/001.mid')