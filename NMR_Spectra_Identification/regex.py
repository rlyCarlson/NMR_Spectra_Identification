import pandas as pd
import re

if __name__ == '__main__':
    input_file = "nmrshiftdb2.xml"
    output_file = "preprocessed.csv"
    ###
    # DATA TO BE COLLECTED
    # - id
    # Inputs:
    #   - X Coordinates of peaks + multiplicity
    #   - Solvent
    #   - Field strength
    #   - Temperature
    #   - NMR type
    # Outputs:
    #   - Molecule Name
    ###
    beta_unicode = '\u03B2'
    alpha_unicode = '\u03B1'
    omega_unicode = '\u03a9'
    lowercase_omega_unicode = '\u03c9'
    delta_unicode = '\u03b4'
    uppercase_delta_unicode = '\u0394'
    gamma_unicode = '\u03b3'
    umlaut_unicode = '\u00f6'
    id_to_name = {} # key is molecule id
    with open(input_file, 'r') as in_f:
        xml_text = in_f.read()
    molecule_texts = re.findall(r'<molecule ([\s\S]*?)</molecule>', xml_text)
    spectrum_texts = re.findall(r'<spectrum ([\s\S]*?)</spectrum>', xml_text)
    print("Number of molecules and spectra found:", len(molecule_texts), len(spectrum_texts))
    # Go through each molecule
    c = 0
    num_mult_name = 0
    for molecule in molecule_texts:
        title = re.findall(r'title="([^"]*)" id', molecule)
        if(len(title) == 0):
            continue
        title = title[0]

        if "Unreported" in title:
            continue
        if alpha_unicode in title:
            title = title.replace(alpha_unicode, 'alpha')
        if beta_unicode in title:
            title = title.replace(beta_unicode, 'beta')
        if gamma_unicode in title:
            title = title.replace(gamma_unicode, 'gamma')
        if delta_unicode in title:
            title = title.replace(delta_unicode, 'delta')
        if omega_unicode in title:
            title = title.replace(omega_unicode, 'omega')
        if lowercase_omega_unicode in title:
            title = title.replace(lowercase_omega_unicode, 'omega')
        if uppercase_delta_unicode in title:
            title = title.replace(uppercase_delta_unicode, 'delta')
        if umlaut_unicode in title:
            title = title.replace(umlaut_unicode, 'o')
        id = re.findall(r'id="([^"]*)">', molecule)
        id_to_name[id[0]] = title # No duplicates in name
    data = {}
    max_length = [-1, None, -1]
    MAX_ALLOWED = 20
    types = set()
    cc = 0
    for spectrum in spectrum_texts:
        # Grab id
        id = re.findall(r'moleculeRef="([^"]*)" type', spectrum)
        if len(id) != 1:
            continue
        id = id[0]
        if id not in id_to_name.keys(): # Name not found, throwing out
            continue
        # Get NMR type
        nmr_type = re.findall(r'<metadata name="nmr:OBSERVENUCLEUS" content="(\d+H|\d+C|\d+F|\d+B|Unreported|NA)"', spectrum)
        if len(nmr_type) == 0:
            tem = re.findall(r'<metadata name="nmr:OBSERVENUCLEUS" content="([a-zA-Z0-9!@#$]+)"', spectrum)
            if len(tem) != 0:
                types.add(tem[0])
            nmr_type = 'NA'
        else:
            nmr_type = nmr_type[0]
            if 'Unreported' in nmr_type:
                print("THERE IS AN UNREPORTED")
                nmr_type = 'NA'
        # Grab temp:
        temp = re.findall(r'dictRef="cml:temp" units="siUnits:k">([a-zA-Z0-9!@#$]+)', spectrum)
        if len(temp) == 0:
            temp = 'NA'
        else:
            temp = temp[0]
            if "Unreported" in temp:
                temp = "NA"
        # Grab field:
        field = re.findall(r'dictRef="cml:field" units="siUnits:megahertz">([a-zA-Z0-9!@#$]+)', spectrum)
        if len(field) == 0:
            field = 'NA'
        else:
            field = field[0]
            if 'Unreported' in field:
                field = "NA"
        # Grab solvent
        solvent = re.findall(r'substance dictRef="cml:solvent" role="subst:solvent" title="([\s\S]*?)"', spectrum)
        solvent_1 = "NA"
        solvent_2 = "NA"
        if len(solvent) != 0:
            solvent = solvent[0]
            if 'Unreported' in solvent or "Unknown" in solvent:
                solvent_1 = "NA"
                solvent_2 = "NA"
            else:
                solvents = re.split(r'\s*(?: or |;|\+)\s*', solvent)
                solvent_1 = solvents[0]
                solvent_2 = solvents[1] if len(solvents) > 1 else 0

        # Grab Peak information
        coordinates = re.findall(r'<peak xValue="(-?\d+(?:\.\d+)?)"', spectrum)
        coordinates_pruned = [float(i) for i in coordinates]
        # Clean it up
        seen = set()
        coordinates_pruned = [x for x in coordinates_pruned if not (x in seen or seen.add(x))]
        sorted_indices = sorted(range(len(coordinates_pruned)), key=lambda i: coordinates_pruned[i])
        coordinates_sorted = [coordinates_pruned[i] for i in sorted_indices]
        if len(coordinates_sorted) > max_length[0]:
            max_length[0] = len(coordinates_sorted)
        if len(coordinates_sorted) > MAX_ALLOWED:
            continue
        # Check if there is a duplicate entry
        best_nmr = ['13C', '1H']
        if id in data.keys():
            name, nmr2, t, f, s1, s2, coord = data[id]
            ### NMR types, if already have a 13C or 1H use that instead
            if nmr2 not in best_nmr and nmr_type in best_nmr:
                print("Better NMR")
                data[id] = (id_to_name[id], nmr_type, temp, field, solvent_1, solvent_2, coordinates_sorted)
            elif (nmr2 in best_nmr and nmr_type in best_nmr) or (nmr2 not in best_nmr and nmr_type not in best_nmr):
                #if counter is negative, dont change, if counter is positive, change
                counter = 0
                if t == "NA" and temp != "NA":
                    counter += 1
                else:
                    counter -= 1
                if f == "NA" and field != "NA":
                    counter += 1
                else:
                    counter -= 1
                if s1 == "NA" and solvent_1 != "NA":
                    counter += 1
                else:
                    counter -= 1
                if len(coord) > len(coordinates_sorted) and len(coordinates_sorted) > 0:
                    counter += 1
                else:
                    counter -= 1
                if counter > 0:
                    cc += 1
                    data[id] = (id_to_name[id], nmr_type, temp, field, solvent_1, solvent_2, coordinates_sorted)
            else:
                # Don't update
                continue
        else:
            data[id] = (id_to_name[id], nmr_type, temp, field, solvent_1, solvent_2, coordinates_sorted)
    print("Number changed", cc)
    # Create Dataframe
    orig = ["ID", "Name", "Type", "Temp", "Field", "Solvent1", "Solvent2"]
    cols = orig.copy()
    for i in range(min(max_length[0], MAX_ALLOWED)):
        cols.append(f"X{i+1}")
    rows = []
    multiplicity_types = set()
    nmr_types = set()
    nmr_mapping = {'1H': 1, '11B': 2, '13C': 3, '19F': 4, 'NA': 5}
    with open('solvent_list.txt', 'r') as sl:
        solvents_seen = sl.readlines()
    solvents_seen = [line.strip() for line in solvents_seen]
    for id in data.keys():
        name, nmr_type, temp, field, solvent_1, solvent_2, coord = data[id]
        nmr_types.add(nmr_type)
        if solvent_1 != 0:
            solvent_1 = (solvents_seen.index(solvent_1.lower()) if solvent_1 != 'NA' else solvents_seen.index(solvent_1)) + 1
        if solvent_2 != 0:
            solvent_2 = (solvents_seen.index(solvent_2.lower()) if solvent_2 != 'NA' else solvents_seen.index(solvent_2)) + 1
        row = {cols[0]: id, cols[1]: name, cols[2]: nmr_mapping[nmr_type], cols[3]: temp, cols[4]: field, cols[5]: solvent_1, cols[6]: solvent_2}
        coord.extend(["NA"] * (min(max_length[0], MAX_ALLOWED) - len(coord)))
        for i in range(len(coord)):
            row[cols[i + len(orig)]] = coord[i]
        rows.append(row)
    print("TOTAL NUMBER PREPROCESSED", len(data.keys()))
    print("NMR types", nmr_types)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv('preprocessed_nmr.csv', index=False)