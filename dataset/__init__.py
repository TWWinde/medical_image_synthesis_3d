import sys
sys.path.append('/misc/no_backups/s1449/medical_image_synthesis_3d')
sys.path.append('/misc/no_backups/s1449')
from dataset.breast_uka import BreastUKA
from dataset.mrnet import MRNetDataset
from dataset.brats import BRATSDataset
from dataset.adni import ADNIDataset
from dataset.duke import DUKEDataset
from dataset.lidc import LIDCDataset
from dataset.default import DEFAULTDataset
from dataset.synthrad2023 import SynthRAD2023Dataset
