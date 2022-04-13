# This file is used for reading out all data
# for profiles, csvs with genotypes, csvs with analyst readings.


import pandas as pd
from classes import *


def txt_read_sample(filename: str):
    """ Function to read data files
    Returns a list of sample names, colors,
    and the data itself as matrix."""
    with open("data/trace_data/" + filename, "r") as text_file:
        texts = text_file.read()
    texts = texts.split("\n")  # split into lines
    # lines 1 and 2 are not interesting
    titles = texts[2].split('\t')  # get titles of files
    titles = [item for item in titles if item != '']  # remove empty entries after splitting
    colors = texts[3].split('\t')  # only needed for width of lines
    data = np.zeros((len(texts[4:]), len(colors)))
    counter = 0  # counter is needed for line number
    for elt in texts[4:]:
        new = np.array(elt.split('\t'))  # split into words
        new[new == ''] = 0
        data[counter, :] = new
        counter += 1
    # now pour contents into separate sample dataclasses
    sample_list = []
    prevname = ""
    replica = 1
    for i in range(len(titles)):
        name = titles[i].split('_')[0]
        if prevname == name:
            replica += 1
        else:
            replica = 1
        new_sample = Sample(name, replica, data[:, 6 * i:6 * i + 6])
        sample_list.append(new_sample)
        prevname = name
    return sample_list


def OLD_txt_read_sample_PROVEDIt(filename: str):
    """ Function to read data from PROVEDIt sized trace data
    almost identical to txt_read_sample, was eventually not used """
    with open("data/trace_data/" + filename, "r") as text_file:
        texts = text_file.read()
    texts = texts.split("\n")  # split into lines
    # lines 1 an 2 are not interesting
    titles = texts[2].split('\t')  # get titles of files
    titles = [item for item in titles if item != '']  # remove empty entries after splitting
    colors = texts[3].split('\t')  # only needed for width of lines
    data = np.zeros((len(texts[4:]), len(colors)))
    counter = 0  # counter is needed for line number
    for elt in texts[4:]:
        new = np.array(elt.split('\t'))  # split into words
        new[new == ''] = 0
        data[counter, :] = new
        counter += 1
    # now pour contents into separate sample dataclasses
    sample_list = []
    for ind in range(len(titles)):
        name = titles[ind]
        if name.__contains__('RD14'):
            new_sample = Sample(name, 0, data[:, 6 * ind:6 * ind + 6])
            sample_list.append(new_sample)
    return sample_list


def csv_read_persons(donor_set):
    """reads all profiles from given donor set (1,2,3,4,5 or 6)"""
    file_name = 'data/donor_profiles/Refs_dataset' + str(donor_set) + '_metYstr.csv'
    with open(file_name) as f:
        first_line = f.readline()
    if "," in first_line:
        file_delimiter = ","
    if ";" in first_line:
        file_delimiter = ";"
    donor_peaks = pd.read_csv(file_name, dtype=str, delimiter=file_delimiter)
    person_list = []  # initialize lists
    alleles = []
    person_name = donor_peaks['SampleName'][0]  # get first donor name
    for index, row in donor_peaks.iterrows():  # iterate over all alleles
        if row[0] != person_name:  # we have arrived at a new person
            # store up to now in Person dataclass, start new list
            person_list.append(Person(person_name, alleles))
            alleles = []
        person_name = row[0]  # first entry is person name
        locus = row[1]  # second entry is locus name
        allele1 = locus + "_" + row[2]  # third entry is first allele
        alleles.append(allele1)
        if not row[3] == "YSTR":
            # Y-STR's are only on Y chromosome, so occur only once
            allele2 = locus + "_" + row[3]  # fourth entry is second allele
            alleles.append(allele2)
    person_list.append(Person(person_name, alleles))
    return person_list


def person_contributions(person_list, number_of_donors: int, mixture_type: str):
    """Calculates relative contributions of each person based on mixture type"""
    # Temporary dict to translate letter to row
    letter_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    mixture_row = letter_to_number[mixture_type]  # type of mixture determines the row
    person_dict = {}  # initialize list and dict
    persons = []
    parts = PICOGRAMS[mixture_row]  # global variables for contributions
    total = TOTAL_PICOGRAMS[mixture_row, number_of_donors - 2]
    for i in range(number_of_donors):
        frac = parts[i] / total / 2  # divide by 2 because 2 alleles per locus
        person_dict[person_list[i].name] = frac  # add fraction to person
        persons.append(person_list[i])
    return person_dict, persons


def OLD_csv_read_persons_PROVEDIt(filename='data/donor_profiles/PROVEDIt_RD14-0003 GF Known Genotypes.csv'):
    """reads all profiles from PROVEDIt file, alternative to csv_read_persons()"""
    donor_alleles = pd.read_csv(filename, dtype=str, delimiter=";")
    person_dict = {}
    for index, row in donor_alleles.iterrows():
        sample_id = str(row['Sample ID'])
        if len(sample_id) == 1:
            sample_id = "0" + sample_id
        locus_allele = []
        for marker in donor_alleles.columns[2:]:
            alleles = str(row[marker]).split(',')
            for allele in alleles:
                if allele != 'nan':
                    locus_allele.append(marker + '_' + allele)
        person = Person(sample_id, locus_allele)
        person_dict[sample_id] = person
    return person_dict


def make_person_mixture(mixture_name):
    """Uses person_contributions and csv_read_persons to create expected peaks in person mixture"""
    donor_set, mixture_type, donor_amount = mixture_name  # can be "1A2" for example
    donor_amount = int(donor_amount)
    person_list = csv_read_persons(donor_set)
    person_fracs, persons = person_contributions(person_list, donor_amount, mixture_type)
    person_mix = PersonMixture(mixture_name, persons, person_fracs)
    return person_mix


def OLD_make_person_mix_PROVEDIt(filename: str):
    """Was eventually not used"""
    donorlist = OLD_read_donors_from_PROVEDIt_filename(filename)
    donordict = OLD_csv_read_persons_PROVEDIt()
    donors = []
    for donor in donorlist:
        donors.append(donordict[donor])
    person_mix = PersonMixture(filename, donors, {})
    return person_mix


def OLD_read_donors_from_PROVEDIt_filename(filename: str):
    """Was eventually not used"""
    donors = filename.split("-")[2].split("d")[0].split("_")
    return donors


def csv_read_analyst(sample_name):
    """Read csv file of analyst's identified alleles returns list of corresponding mixtures of peaks"""
    file_name = "data/analysts_data_filtered/" + str(sample_name) + "_New.csv"
    with open(file_name) as f:
        first_line = f.readline()
    if "," in first_line:
        file_delimiter = ","
    if ";" in first_line:
        file_delimiter = ";"
    results = pd.read_csv(file_name, dtype=str, delimiter=file_delimiter)
    name = results['Sample Name'][0]  # to start iteration
    sample_name, replicate = name.split('.')
    mixture_list = []  # initialize big lists
    peak_list = []  # initialize small lists
    number_of_columns = int((len(results.columns) - 2) / 2)
    for index, row in results.iterrows():
        # iterate over all rows, because each row contains the peaks for one allele
        if name != row[0]:  # then start new sample
            sample_name, replicate = name.split('.')
            mixture_list.append(AnalystMixture(sample_name, replicate, peak_list))
            peak_list = []  # empty list
        name = row[0]  # then set name to current sample name
        for i in range(2, 2 + number_of_columns):
            # go over the 8-10 possible locations of peak identification
            if str(row[i]) == row[i]:
                # append value only if non-empty
                # empty entries are converted to (float-type) NaN's by pandas
                # so str(row[i]) == row[i] filters out empty entries
                locus = locus_dict[row[1]]
                allele = locus.alleles[row[i]]
                height = float(row[i + number_of_columns])  # heights are 10 indices further than
                new_peak = Peak(allele, height)
                peak_list.append(new_peak)
    mixture_list.append(AnalystMixture(name, replicate, peak_list))
    return mixture_list


def shallow_analyst(sample_name):
    """Read csv file of analyst's identified alleles returns shallow list"""
    file_name = "data/analysts_data_filtered/" + str(sample_name) + "_New.csv"
    with open(file_name) as f:
        first_line = f.readline()
    if "," in first_line:
        file_delimiter = ","
    if ";" in first_line:
        file_delimiter = ";"
    results = pd.read_csv(file_name, dtype=str, delimiter=file_delimiter)
    name = results['Sample Name'][0]
    mixture_list = []
    peak_list = []
    number_of_columns = int((len(results.columns) - 2) / 2)
    for index, row in results.iterrows():
        # iterate over all rows, because each row contains the peaks for one allele
        if name != row[0]:  # start new sample
            mixture_list.append(peak_list)
            peak_list = []  # empty list
        name = row[0]  # then set name to current sample name
        for i in range(2, 2 + number_of_columns):
            # go over the 8-10 possible locations of peak identification
            if str(row[i]) == row[i]:
                # append value only if non-empty
                locus = row[1]
                allele = row[i]
                peak_list.append(str(locus) + "_" + str(allele))
    mixture_list.append(peak_list)
    return mixture_list


def read_csv_to_dataframe(filename, loc):
    return pd.read_csv(loc+filename)