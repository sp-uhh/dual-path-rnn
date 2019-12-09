# This file holds general utility functions

def text_file_2_list(text_file):
    with open(text_file) as f:
        list_of_strings = f.readlines()
    list_of_strings = [x.strip() for x in list_of_strings]
    return list_of_strings
