import sys
sys.path.append('/misc/no_backups/s1449/medical_image_synthesis_3d')
sys.path.append('/misc/no_backups/s1449')
from dataset_.breast_uka import BreastUKA
from dataset_.mrnet import MRNetDataset
from dataset_.brats import BRATSDataset
from dataset_.adni import ADNIDataset
from dataset_.duke import DUKEDataset
from dataset_.lidc import LIDCDataset
from dataset_.default import DEFAULTDataset
from dataset_.synthrad2023 import SynthRAD2023Dataset
