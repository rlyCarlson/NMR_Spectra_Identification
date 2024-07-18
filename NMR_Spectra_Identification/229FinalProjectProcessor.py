from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen
from urllib.parse import quote
from rdkit import Chem
import pandas as pd
import random
# convert to a readable form
def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        print(ids)
        return 'Did not work'
# check for the case where we have a carbonyl becausse there are many combinations of carbonyls so we need to identify this first
def neighbor_has_carbonyl(neighbor):
    if neighbor.GetSymbol() != 'C':
        return False
    # check if the carbon atom has any double bonds
    for bond in neighbor.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            # check if the other atom in the bond is oxygen
            other_atom = bond.GetOtherAtom(neighbor)
            # existence of carbonyl group C = O
            if other_atom.GetSymbol() == 'O':
                return True  
    return False

def classify_functional_groups(smiles):
    # convert smiles string to readable format
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(smiles)
        return -1

    # all the functional groups
    functional_groups = {
        "Alkane": '[CX4][!#6]',
        "Alkene": '[CX3]=[CX3]',
        "Alkyne": '[CX2]#[CX2]',
        "Carbonyl": '[#6]=[#8]',
        "Ester": '[CX3](=[OX1])[OX2][#6]',
        "Amide": '[#6][CX3](=[OX1])[NX3;H2,H1]',
        "Amine": '[NX3;H2,H1][#6]',
        "Aromatic Ring": '[c]',
        "Nitrile": '[NX1]#[CX2]',
        "Alcohol": '[OX2H1]',
        "Carboxylic Acid": 'C(=O)O',
        "Ether": 'COC'
    }
    leftToCheck = {
        "Alkane": '[CX4]',
        "Alkene": '[CX3]=[CX3]',
        "Alkyne": '[CX2]#[CX2]',
        "Aromatic Ring": '[c]',
    }
    # dictionary to store the presence of each functional group
    presence = {group: mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for group, pattern in leftToCheck.items()}

    # iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol != 'C':
            continue
        # go through and find the group bonded to the carbon
        neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O']
        nitrogens = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'N']
        aminos = 0
        nitros = 0
        alc = sum(1 for neighbor in neighbors if neighbor.GetNumImplicitHs() == 1)
        carbonyl = 1 if any((len(neighbor.GetNeighbors()) == 1 and neighbor.GetNumImplicitHs() == 0)for neighbor in neighbors) else 0
        ester = 1 if any((len(neighbor.GetNeighbors()) == 2 and neighbor.GetNumImplicitHs() == 0 and any(neighbor_has_carbonyl(ne) for ne in neighbor.GetNeighbors()))for neighbor in neighbors) else 0
        ether = 1 if any((len(neighbor.GetNeighbors()) == 2 and neighbor.GetNumImplicitHs() == 0 and not any(neighbor_has_carbonyl(ne) for ne in neighbor.GetNeighbors()))for neighbor in neighbors) else 0
        for elem in nitrogens:
            CN = mol.GetBondBetweenAtoms(atom.GetIdx(), elem.GetIdx())
            order = CN.GetBondTypeAsDouble()
            if order == 1:
                aminos = 1
            if order == 3:
                nitros = 1
        if alc >= 1 and carbonyl == 0:
            presence["Alcohol"] = True
        if alc >= 1 and carbonyl == 1:
            presence["Carboxylic Acid"] = True
        if carbonyl == 1 and ester >= 1:
            presence["Ester"] = True
        if carbonyl == 0 and ether >=1:
            presence["Ether"] = True
        if carbonyl == 1 and ether == 0 and ester == 0 and alc == 0 and aminos == 0:
            presence["Carbonyl"] = True
        if aminos == 1 and carbonyl == 0:
            presence["Amine"] = True
        if aminos == 1 and carbonyl == 1:
            presence["Amide"] = True
        if nitros == 1:
            presence["Nitrile"] = True

    classification = [group for group, present in presence.items() if present]
    return classification
    
def process_identifier(elem):
    elems = elem.split(';')
    newElems = sorted(elems, key=len)
    valid = False
    for e in newElems:
        val = CIRconvert(e.strip())
        if val == 'Did not work':
            continue
        classification = classify_functional_groups(val)
        if classification != -1:
            valid = True
            break
    if not valid:
        return None
    return classification
# read in the spreadsheet in question
path = 'bad_nmr_preproccesed.csv'
df = pd.read_csv(path)
df_bad = pd.DataFrame(columns=df.columns)
fxn = ["Alkane", "Alkene", "Alkyne", "Carbonyl", "Ester", "Amide", "Amine", "Aromatic Ring", "Nitrile", "Alcohol", "Carboxylic Acid", "Ether"]
for group in fxn:
    df[group] = 0
identifiers = df['Name'].tolist()
# mutlithread to speed up run time
with ThreadPoolExecutor(max_workers=12) as executor:
    results = executor.map(process_identifier, identifiers)

rows_to_drop = []
for elem_i, classification in enumerate(results):
    if classification is None:
        rows_to_drop.append(elem_i)
        df2 = df.iloc[[elem_i]]
        df_bad = pd.concat([df_bad, df2], ignore_index=True)
        print("dropped " + str(elem_i))
        continue
    for group in fxn:
        if group in classification:
            df.loc[elem_i, group] = 1
print(len(rows_to_drop))
df = df.drop(rows_to_drop)
print(df.iloc[:, -12:])
df.to_csv('bad_nmr_proccesed.csv', index=False)
df_bad.to_csv('bad_bad.csv', index=False)