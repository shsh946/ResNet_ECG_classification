# ResNet_ECG_classification
Use 1-dimension ResNet to classify ECG pulses.


Training with MITBIH dataset with 7 classes.

    ‘/’ : Paced beat
    ‘V’: Premature ventricular contraction
    ‘N’: Normal beat
    ‘A’ : Atrial premature beat
    ‘R’ : Right bundle branch block beat
    ‘L’ : Left bundle branch block beat
    '[]': Atrial fibrillation

Every data is a pulse with its label contain 512 sample points before and after, so there will be 1024 points in it.

- Python 3.7
- Pytorch 1.2(stable)

