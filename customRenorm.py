# Copied from that one notebook where I first introduced this concept.
# Specifically, this: https://colab.research.google.com/drive/1yz2xsidgWaCqoAbJCMePF376jEhYh0k1


import math
from scipy.stats import norm

# First portion is just simple estimates; I can write this more professionally for the final draft.

def maximalMu(M, N):
  return norm.ppf(1 - (1 / M))*math.sqrt(N)

def maximalSig(M, N):
  return (norm.ppf(1 - (1 / (M*math.e))) - norm.ppf(1 - (1 / M)))*math.sqrt(N)

# WARNING: table incomplete at http://faculty.washington.edu/heagerty/Books/Biostatistics/TABLES/NormalOrder.pdf and https://projecteuclid.org/download/pdf_1/euclid.aoms/1177728266
# Using taylor series-based approximations for 25, 26, and 27
Mmus = [0, 0.56419, 0.84628, 1.02938, 1.16296,\
        1.26721, 1.35218, 1.42360, 1.48501, 1.53875,\
        1.58644, 1.62923, 1.66799, 1.70338, 1.73591,\
        1.76599, 1.79394, 1.82003, 1.84448, 1.86748,\
        1.88917, 1.90969, 1.92916, 1.94767, 1.94767 + 0.0177,
        1.94767 + 0.0177 + 0.0169,\
        1.94767 + 0.0177 + 0.0169 + 0.0161,\
                           2.01371, 2.02852, 2.04276,\
        2.05646, 2.06967, 2.08241, 2.09471, 2.10661,\
        2.11812, 2.12928]

# And the Msig, coming from this table: https://projecteuclid.org/download/pdf_1/euclid.aoms/1177728266 (Table II)
# TableII is the expectation of x^2; as usuall, the squared mean can be subtraced for the variance.
TableII = [1.0, 1.0, 1.27566, 1.55132, 1.80002,\
           2.02174, 2.22030, 2.39953, 2.56262, 2.71210,\
           2.85003, 2.97802, 3.09740, 3.20924, 3.31444,\
           3.41374, 3.50766, 3.59705, 3.68205, 3.76316]

Mvar = [TableII[i] - (Mmus[i]**2) for i in range(min(len(TableII), len(Mmus)))]

Msig = [math.sqrt(x) for x in Mvar]

trueMuDif = Mmus[-1] - maximalMu(len(Mmus), 1)
trueSigDif = Msig[-1] - maximalSig(len(Msig), 1)

#Which Leads to this hacky 'true' func:
# Replace me for final draft with a full table lookup
def trueMaximalStats(M, N):
  scale = math.sqrt(N)

  if M <= len(Mmus):
    mu = Mmus[M - 1]
  else:
    mu = maximalMu(M, 1) + trueMuDif
  if M <= len(Msig):
    sig = Msig[M - 1]
  else:
    sig = maximalSig(M, 1) + trueSigDif

  return scale*mu, scale*sig

# This function has a "hacky" feel, but it works incredibly well, as I showed in the Colab notebook using Monte Carlo testing for a range of values of M.
# OK, now on to the actual renormalization classes. This part doesn't have to change for the final draft

import numpy as np

import tensorflow as tf
import larq as lq

from tensorflow.keras.models import Model

# N in both is the number of things from the last stage added together.
# N = 1 for properly-initialized full-precision layers
# N = fan_in for binary layers.

# Uses Gaussian order statistics to make mu = 0 and sigma = 1 after a MaxPool operation.
class MaxpoolRenorm(Model): 
  def __init__(self, M, N, origMu = 0.0, origSig=1.0, customOffset = 0.0): # origSig is sigma of input layer. Same for origMu 
    super(MaxpoolRenorm, self).__init__()
    self.mu, self.sig = trueMaximalStats(M, N)
    self.mu = origMu + self.mu * origSig
    self.sig *= origSig
    self.customOffset = customOffset

  def call(self, x):
    return (x - self.mu) / self.sig + self.customOffset


# Distribution is 0.5*(delta function at 0) + 0.5*(half-normal distribution).
# Half-normal distribution: https://en.wikipedia.org/wiki/Half-normal_distribution
# Monte Carlo tested.
class ReluNormalRenorm(Model): 
  """Specially designed for the mu and sigma of relu(normal_sample). origMu must be 0.0."""
  def __init__(self, N, origSig = 1.0, customOffset = 0.0):
    super(ReluNormalRenorm, self).__init__()
    self.mu = origSig * np.sqrt(0.5*N/np.pi) 
    self.sig = origSig * np.sqrt(0.5*N - (0.5*N / np.pi)) 
    self.customOffset = customOffset

  def call(self, x):
    return (x - self.mu) / self.sig + self.customOffset

# EXPERIMENTAL.
# This one is not complete yet, and requires further debugging.
class BlurpoolRenorm(Model): # Must also use MaxpoolRenorm after the maxpool.
  def __init__(self, strides):
    super(BlurpoolRenorm, self).__init__()
    if strides == 2:
      self.sig = 9.12 / 16
    elif strides == 3:
      self.sig = 11.33 / 16
    elif strides == 4:
      self.sig = 12.32
    elif strides == 5:
      self.sig == 12.89 / 16
    else:
      raise NotImplementedError("Bad Strides")

  def call(self, x):
    return x / self.sig




