# Machine Learning based Estimation of Aortic Pressure Curves by Electrical Impedance Tomography

Abstract:
Central aortic pressure is a key hemodynamic
parameter to monitor and target in clinical practice. As this
gold standard method is highly invasive and conventional non-
invasive methods are not long-term compatible or inaccurate,
the need for alternative monitoring capabilities arises. Electrical
impedance tomography (EIT) is a non-invasive monitoring
technique using an electrode belt around the torso. In this
paper, EIT recordings from an in-vivo animal model and
simultaneously recorded central aortic pressure measurements
from invasive catheters are used to train a convolutional neural
network predicting aortic pressure curves from EIT voltages.
Different parametric representations of the aortic pressure
time series are considered to reduce network complexity. A
hyperparameter tuning is conducted to optimize the network.
Results demonstrate that the estimation of aortic pressure
curves by a trained network is feasible even on unknown test
data, however, random offsets are observed.




This repository contains algorithms, training routines and results for EIT based estimation of central aortic pressure (CAP) curves

#
