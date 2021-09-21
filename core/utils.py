import os
import numpy as np
import re
import tarfile
import shutil

letter = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'
others = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
alphabet = letter + digits + others
print('alphabet size:', len(alphabet))

# all-zeroes padding vector:
pad_vector = [0 for x in alphabet]

# pre-calculated one-hot vectors:
supported_chars_map = {}

for i, ch in enumerate(alphabet):
  vec = [0 for x in alphabet]
  vec[i] = 1
  supported_chars_map[ch] = vec

#print('one-hot encoding for character c:', supported_chars_map['c'])
langs = [
  "C",
  "C#",
  "C++",
  "D",
  "Haskell",
  "Java",
  "JavaScript",
  "PHP",
  "Python",
  "Rust"
]

num_classes = len(langs)

def get_source_snippets(file_name, breakup=False):
  # Read the file content and lower-case:                                    
  text = ""
  print(type(file_name))
  with open(file_name, mode='r') as file:
    text = file.read().lower()
  lines = text.split('\n')
  nlines = len(lines)
  if breakup and nlines > 50:
    aThird = nlines//3
    twoThirds = 2*aThird
    text1 = '\n'.join(lines[:aThird])
    text2 = '\n'.join(lines[aThird:twoThirds])
    text3 = '\n'.join(lines[twoThirds:])
    return [text1, text2, text3]
  return [text]

def turn_sample_to_vector(sample, sample_vectors_size=1024,
                          normalize_whitespace=True):
  if normalize_whitespace:
    # Map (most) white-space to space and compact to single one:
    sample = sample.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    sample = re.sub('\s+', ' ', sample)

  # Encode the characters to one-hot vectors:
  sample_vectors = []
  for ch in sample:
    if ch in supported_chars_map:
      sample_vectors.append(supported_chars_map[ch])

  # Truncate to fixed length:
  sample_vectors = sample_vectors[0:sample_vectors_size]

  # Pad with 0 vectors:
  if len(sample_vectors) < sample_vectors_size:
    for i in range(0, sample_vectors_size - len(sample_vectors)):
      sample_vectors.append(pad_vector)

  return np.array(sample_vectors)

def turn_file_to_vectors(file_name, sample_vectors_size=1024,
                         normalize_whitespace=True, breakup=False):
  samples = get_source_snippets(file_name, breakup)
  return [turn_sample_to_vector(s, sample_vectors_size, normalize_whitespace)
          for s in samples]
