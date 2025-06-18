import numpy as np

labels_ = np.array(['H/D','HE','LI','BE','B','C','N','O','F','NE','NA','MG','AL','SI','P','S',
        'CL','AR','K','CA','SC','TI','V','CR','MN','FE','CO','NI','CU','ZN','GA','GE','AS',
        'SE','BR','KR','RB','SR','Y','ZR','NB','MO','TC','RU','RH','PD','AG','CD','IN','SN',
        'SB','TE','I','XE','CS','BA','LA','CE','PR','ND','PM','SM','EU','GD','TB','DY','HO',
        'ER','TM','YB','LU','HF','TA','W','RE','OS','IR','PT','AU','HG','TL','PB','BI','PO',
        'AT','RN','FR','RA','AC','TH','PA','U','NP','PU','AM','CM','BK','CF','ES','FM','MD',
        'NO','LR','RF','DB','SG','BH','HS','MT','DS','RG','CP','UUT','UUQ','UUP','UUUH','UUS','UUO'])

def get_atom_index(pAtomType):
    """Method to get the index of the atom based on its type.

    Args:
        pAtomType (string): Atom label.

    Returns:
        int: Index in the list of the atom type.
    """

    index = -1
    for it, atom in enumerate(labels_):
        currALabelSplit = atom.split("/")
        for auxCurrLabel in currALabelSplit:
            if pAtomType == auxCurrLabel:
                index = it

    return index


aLabels_ = np.array([
        'HIS/HID/HIE/HIP', #0
        'ASP/ASH', #1
        'ARG/ARN', #2
        'PHE', #3
        'ALA', #4
        'CYS/CYX', #5
        'GLY', #6
        'GLN', #7
        'GLU/GLH', #8
        'LYS/LYN', #9
        'LEU', #10
        'MET', #11
        'ASN', #12
        'SER', #13
        'TYR', #14
        'THR', #15
        'ILE', #16
        'TRP', #17
        'PRO', #18
        'VAL', #19
        'SEC', #20
        'PYL', #21
        'ASX', #22
        'XLE', #23
        'GLX', #24
        'XXX']) #25

def get_aminoacid_index(pALabel):
    """Method to get the index of the aminoacid based on its label.

    Args:
        pALabel (string): Aminoacid label.

    Returns:
        int: Index in the list of the Aminoacid type.
    """

    index = -1
    for it, currALabel in enumerate(aLabels_):
        currALabelSplit = currALabel.split("/")
        for auxCurrLabel in currALabelSplit:
            if pALabel == auxCurrLabel:
                index = it
    return index