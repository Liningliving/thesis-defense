import json
import ast

def remove_image_triples(triples):
    if not isinstance(triples, list):
        # leave None or other types untouched
        return triples
    # keep only those triples where none of the parts contain "image"
    return [
        triple for triple in triples
        if not any("image" in part for part in triple)
    ]


def clean_triples(triples):
    # If it's not a list (e.g. None), leave it alone
    if not isinstance(triples, list):
        return triples
    cleaned = []
    for triple in triples:
        # filter out any empty strings from the inner list
        filtered = [part for part in triple if part != ""]
        # (Optionally) only keep lists of length exactly 3 afterwards
        # if len(filtered) == 3:
        cleaned.append(filtered)
    return cleaned

def delete_image_from_list(relations_list):
    if relations_list == []:
        return []
    
    for relation in relations_list:
        for item in relation:
            if "image" in item:
                relations_list.remove(relation)
    return relations_list