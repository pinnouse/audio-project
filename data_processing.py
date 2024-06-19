import array
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, List, Optional, Tuple

import numpy as np

# https://github.com/mcdermottLab/pycochleagram
import pycochleagram.cochleagram as cgram
from pydub import AudioSegment
from scipy import ndimage

DATA_DIR = os.path.join("english")

class PronunciationTiming(object):

    def __init__(self, word: str, start: int, end: int) -> None:
        self.word = word
        self.start = start
        self.end = end

class SWC(object):

    def __init__(self, swc_path: str) -> None:
        self.tree = ET.parse(swc_path)
        self.timings = []

    def get_timings(self) -> List[PronunciationTiming]:
        if len(self.timings) > 0:
            return self.timings
        # https://www.geeksforgeeks.org/xml-parsing-python/
        root = self.tree.getroot()
        doc = root.find('d')

        for p in doc.findall('p'):
            for s in p.findall('s'):
                for t in s.findall('t'):
                    for n in t.findall('n'):
                        attribs = n.attrib
                        if 'pronunciation' not in attribs \
                            or 'start' not in attribs or 'end' not in attribs:
                            continue
                        word = attribs['pronunciation'].lower()
                        self.timings.append(PronunciationTiming(
                            word,
                            int(attribs['start']),
                            int(attribs['end'])))
        return self.timings


def split_clip(audio: AudioSegment, swc: SWC
               ) -> Tuple[List[str], List[AudioSegment]]:
    words = []
    clips = []
    for p in swc.get_timings():
        words.append(p.word)
        clip = audio[p.start:p.end]
        # https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
        # clip = audio[p.start:p.end].get_array_of_samples()
        # fp_arr = np.array(clip).T.astype(np.float32)
        # fp_arr /= np.iinfo(clip.typecode).max
        clips.append(clip)
    return words, clips

class PronunciationClip(object):
    def __init__(self, word: str, audio: AudioSegment) -> None:
        self.word = word
        self.audio = audio

    def __repr__(self) -> str:
        return f"PronunciationClip({repr(self.word)}, {repr(self.audio)})"

def make_pronunciation_clip(path: os.PathLike, vocab: List[str] = None,
                            i: int = 0, starttime: float = 0
                            ) -> List[PronunciationClip]:
    if (i + 1) % 50 == 0:
        print(f'Processing file #{i+1}, took {time.time() - starttime}s')
    audio = AudioSegment.from_ogg(os.path.join(DATA_DIR, path, 'audio.ogg'))
    swc = SWC(os.path.join(DATA_DIR, path, 'aligned.swc'))
    words, pcms = split_clip(audio, swc)
    clips = []
    for w, c in zip(words, pcms):
        if len(w) < 4 \
            or vocab is not None and w not in vocab:
            continue
        clips.append(PronunciationClip(w, c))
    return clips

def load_utterances(limit_files: Optional[int] = 500, vocab: List[str] = None
                    ) -> None:
    def verify_dir(d: os.PathLike) -> bool:
        path = os.path.join(DATA_DIR, d)
        if not os.path.isdir(path):
            return False
        files = os.listdir(path)
        return 'audio.ogg' in files and 'aligned.swc' in files
    path = os.path.join('parsed_files', 'raw_clips')
    utter_path = os.path.join('parsed_files', 'raw_utterance')
    os.makedirs(path, exist_ok = True)
    os.makedirs(utter_path, exist_ok = True)

    files = list(filter(verify_dir, os.listdir(DATA_DIR)))
    if limit_files is not None:
        files = files[:limit_files]
    starttime = time.time()
    futures = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(make_pronunciation_clip,
                                   f,
                                   vocab,
                                   i,
                                   starttime): f for i, f in enumerate(files)}
        i = 0
        for future in concurrent.futures.as_completed(futures):
            c = futures[future]
            try:
                for c in future.result():
                    og = c.audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(4)
                    og.export(os.path.join(utter_path, f'{c.word}_{i}.wav'), format="wav")
                    a = gen_clip(c.audio)
                    a.export(os.path.join(path, f'{c.word}_{i}.wav'), format="wav")
                    i += 1
            except Exception as e:
                print(f'Failed to parse file: {c} ({e})')

def make_noise_playlist(path: os.PathLike, limit: int = 0) -> List[AudioSegment]:
    clips = []
    for f in os.listdir(path):
        if not f.endswith('wav'):
            continue
        file_path = os.path.join(path, f)
        clips.append(AudioSegment.from_file(file_path))
        if limit > 0 and len(clips) > limit:
            break
    return clips



N_FILTERS = 50
LO_LIM = 30
HI_LIM = 7860
SAMPLE_RATE = 16000

def gen_clip(audio: AudioSegment, duration: int = 2000) -> AudioSegment:
    clip = AudioSegment.silent(duration=duration, frame_rate=SAMPLE_RATE)
    audio_duration = len(audio)
    offset = max(0, duration / 2 - (audio_duration / 2))
    return clip.overlay(audio, offset)

def audio_to_cgram(audio: AudioSegment) -> np.array:
    # TODO: find audio input type for cochleagram (8-bit integer PCM?)
    arr = np.array(audio.get_array_of_samples())
    return cgram.human_cochleagram(arr, audio.frame_rate or SAMPLE_RATE,
                                   N_FILTERS, LO_LIM, HI_LIM, 4, downsample=200)

def aa_cgram(cochleagram: np.ndarray, size: int = 256) -> np.ndarray:
    f, t = cochleagram.shape
    kx = size / f
    ky = size / t
    return ndimage.zoom(cochleagram, (kx, ky))


def make_word2ind(clips: List[PronunciationClip]) -> Dict[str, int]:
    words2ind = {}
    words = set(c.word for c in clips)
    for i, w in enumerate(words):
        words2ind[w] = i
    return words2ind

def load_vocab(vocab_file: str = 'vocab.txt') -> List[str]:
    vocab = []
    with open(vocab_file, 'r') as f:
        vocab = f.read().strip().split('\n')
    print(f'Loaded vocabulary {vocab_file} with {len(vocab)} words.')
    return vocab

def vocab_word2ind(vocab: List[str]) -> Dict[str, int]:
    x = {}
    for i, w in enumerate(vocab):
        x[w] = i
    return x

def filter_clips_vocab(clips: List[PronunciationClip], vocab: List[str]
                 ) -> List[PronunciationClip]:
    return list(filter(lambda c: c.word in vocab, clips))

def load_clips(path: str = 'parsed_files/raw_clips') -> List[PronunciationClip]:
    clips = []
    for fp in os.listdir(path):
        if not fp.endswith('.wav'):
            continue
        filename = os.path.join(path, fp)
        audio = AudioSegment.from_file(filename)
        word = fp.split('_')[0]
        clips.append(PronunciationClip(word, audio))
    return clips

def overlay_noise(snr: float, signal: AudioSegment, noise: AudioSegment
                  ) -> AudioSegment:
    noise = noise.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(4)
    signal = signal.set_frame_rate(SAMPLE_RATE)
    signal_rms = signal.rms
    noise_rms = noise.rms
    scaling_factor = (signal_rms / noise_rms) * (10**(-snr/20))
    noise_np = np.array(noise.get_array_of_samples())
    noise_samples = noise_np * scaling_factor
    noise_array = array.array(signal.array_type, np.round(noise_samples).astype(np.int32))
    scaled_noise = noise._spawn(noise_array)
    return signal.overlay(scaled_noise, loop=True)

def filter_clips(clips: List[PronunciationClip], min_freq: int = 20,
                 max_freq: int = 200) -> List[PronunciationClip]:
    freq_dict = {}
    for pc in clips:
        freq_dict[pc.word] = freq_dict.get(pc.word, 0) + 1
    included_words = list(filter(lambda w: min_freq < freq_dict[w] < max_freq,
                                 freq_dict.keys()))
    return list(filter(lambda c: c.word in included_words, clips))

def overlay_noise_clips(clip: PronunciationClip, noise: List[AudioSegment],
                        SNRs: List[float], paths: List[os.PathLike],
                        i: Optional[int], starttime: Optional[float]) -> None:
    a = gen_clip(clip.audio)
    if (i + 1) % 200 == 0:
        print(f'Parsing {i+1}-th file, {time.time() - starttime}s since start.')
    for n, snr, path in zip(noise, SNRs, paths):
        noisy_clip = overlay_noise(snr, a, n)
        noisy_cgram = aa_cgram(audio_to_cgram(noisy_clip))

        fname = f'{clip.word}_{i}' if i is not None else clip.word

        noisy_clip.export(os.path.join(path, fname + '.wav'), format="wav")
        np.save(os.path.join(path, fname + '.npy'), noisy_cgram)

def load_cochleagrams(path: os.PathLike, word2ind: Dict[str, int]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    data = []
    targets = []
    for f in os.listdir(path):
        w = f.split('_')[0]
        if not f.endswith(".npy") or w not in word2ind:
            continue
        x = np.load(os.path.join(path, f))
        data.append(x)
        targets.append(word2ind[w])
    return np.array(data, dtype=np.float32), np.array(targets, dtype=np.int64)