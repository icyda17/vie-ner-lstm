from collections import Counter
import json

def is_bio_scheme(all_tags):
    """
    Check if BIO tagging scheme is used. Return True if so.
    Args:
        all_tags: a list of NER tags
    Returns:
        True if the tagging scheme is BIO, otherwise False
    """
    for tag in all_tags:
        if tag == 'O':
            continue
        elif len(tag) > 2 and tag[:2] in ('B-', 'I-'):
            continue
        else:
            return False
    return True

def to_bio2(tags):
    """Convert the original tag sequence to BIO2 format.
    ref: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

    If the input is already in BIO2 format,
    the original input is returned.
    Args:
        tags: a list of tags in either BIO or BIO2 format
        tags: a list of tags in either BIO or BIO2 format
    Args:
        tags: a list of tags in either BIO or BIO2 format
    Returns:
        new_tags: a list of tags in BIO2 format
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag[0] == 'I':
            if i == 0 or tags[i-1] == 'O' or tags[i-1][1:] != tag[1:]:
                new_tags.append('B' + tag[1:])
            else:
                new_tags.append(tag)
    return new_tags

def bio2_to_bioes(tags):
    """Convert the original tag sequence into a BIOES sequence.

    Args:
        tags: a list of tags in original format
    Returns:: a list of tags in BIO2 format
    Returns:: a list of tags in BIO2 format
    Returns:
        new_tags: a list of tags in BIOES format
    """
    tags = to_bio2(tags)
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        else:
            if len(tag) < 2:
                raise Exception(f"Invalid BIO2 tag found: {tag}")
            else:
                if tag[:2] == 'I-':  # convert to E- if next tag is not I-
                    if i+1 < len(tags) and tags[i+1][:2] == 'I-':
                        new_tags.append(tag)
                    else:
                        new_tags.append('E-' + tag[2:])
                elif tag[:2] == 'B-':  # convert to S- if next tag is not I-
                    if i+1 < len(tags) and tags[i+1][:2] == 'I-':
                        new_tags.append(tag)
                    else:
                        new_tags.append('S-' + tag[2:])
    return new_tags

def decode_from_bioes(tags):
    """Decode from a sequence of BIOES tags, assuming default tag is 'O'.

    Args:
        tags: a list of BIOES tags
    Returns:: a list of BIOES tags
    Returns:: a list of BIOES tags
    Returns:
        A list of dict with start_idx, end_idx, and type values.
    """
    res = []
    ent_idxs = []
    cur_type = None

    def flush():
        if len(ent_idxs) > 0:
            res.append({
                'start': ent_idxs[0],
                'end': ent_idxs[-1],
                'type': cur_type})

    for idx, tag in enumerate(tags):
        if tag is None:
            tag = 'O'
        if tag == 'O':
            flush()
            ent_idxs = []
        elif tag.startswith('B-'):  # start of new ent
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
        elif tag.startswith('I-'):  # continue last ent
            ent_idxs.append(idx)
            cur_type = tag[2:]
        elif tag.startswith('E-'):  # end last ent
            ent_idxs.append(idx)
            cur_type = tag[2:]
            flush()
            ent_idxs = []
        elif tag.startswith('S-'):  # start single word ent
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
            flush()
            ent_idxs = []
    # flush after whole sentence
    flush()
    return res

def score_by_entity(pred_tag_sequences, gold_tag_sequences, verbose=True):
    """ Score predicted tags at the entity level.
    Args:
        pred_tags_sequences: a list of list of predicted tags for each word
        gold_tags_sequences: a list of list of gold tags for each word
        verbose: print log with results
    
    Returns:
        Precision, recall and F1 scores.
    """
    assert(len(gold_tag_sequences) == len(pred_tag_sequences)), \
        "Number of predicted tag sequences does not match gold sequences."
    
    def decode_all(tag_sequences):
        # decode from all sequences, each sequence with a unique id
        ents = []
        for sent_id, tags in enumerate(tag_sequences):
            for ent in decode_from_bioes(tags):
                ent['sent_id'] = sent_id
                ents += [ent]
        return ents
    if is_bio_scheme(gold_tag_sequences):
        gold_tag_sequences = bio2_to_bioes(gold_tag_sequences)
    if is_bio_scheme(pred_tag_sequences): 
        pred_tag_sequences = bio2_to_bioes(pred_tag_sequences)
    gold_ents = decode_all(gold_tag_sequences)
    pred_ents = decode_all(pred_tag_sequences)

    # scoring
    correct_by_type = Counter()
    guessed_by_type = Counter()
    gold_by_type = Counter()

    for p in pred_ents:
        guessed_by_type[p['type']] += 1
        if p in gold_ents:
            correct_by_type[p['type']] += 1
    for g in gold_ents:
        gold_by_type[g['type']] += 1
    
    prec_micro = 0.0
    if sum(guessed_by_type.values()) > 0:
        prec_micro = sum(correct_by_type.values()) * 1.0 / sum(guessed_by_type.values())
    rec_micro = 0.0
    if sum(gold_by_type.values()) > 0:
        rec_micro = sum(correct_by_type.values()) * 1.0 / sum(gold_by_type.values())
    f_micro = 0.0
    if prec_micro + rec_micro > 0:
        f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro)
    overall_score = {}
    overall_score['Recall'] = rec_micro
    overall_score['Precision'] = prec_micro
    overall_score['F1'] = f_micro

    def score_tag(tag):
        out = {}
        prec_micro = 0.0
        if guessed_by_type[tag] > 0:
            prec_micro = correct_by_type[tag] * 1.0 / guessed_by_type[tag]
        rec_micro = 0.0
        if gold_by_type[tag] > 0:
            rec_micro = correct_by_type[tag] * 1.0 / gold_by_type[tag]
        f_micro = 0.0
        if prec_micro + rec_micro > 0:
            f_micro = 2.0 * prec_micro * rec_micro / (prec_micro + rec_micro) 
        out['Recall'] = rec_micro
        out['Precision'] = prec_micro
        out['F1'] = f_micro
        return out

    res = {'PER':{},'LOC':{}, 'ORG':{}, 'MISC':{}, 'OVERALL':{}}
    res['PER'] = score_tag('PER')
    res['LOC'] = score_tag('LOC')
    res['ORG'] = score_tag('ORG')
    res['MISC'] = score_tag('MISC')
    res['OVERALL'] = overall_score
    '''
    if verbose:
        logger.info("Prec.\tRec.\tF1")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}".format( \
            prec_micro*100, rec_micro*100, f_micro*100))
    '''
    return res

def read_result(input_file):
    with open(input_file, 'r') as f:
        pred_list = []
        gold_list = []
        preds = []
        golds = []
        for line in f:
            line = line.split()
            if len(line) > 0:
                preds.append(line[1])
                golds.append(line[2])
            else:
                pred_list.append(preds)
                gold_list.append(golds)
                preds = []
                golds = []
                tags = []
    return pred_list, gold_list

def test(pred_sequences, gold_sequences):
    print (json.dumps(score_by_entity(pred_sequences, gold_sequences), indent=2))

def test_file(input_file):
    pred_list, gold_list = read_result(input_file)
    test(pred_list, gold_list)

if __name__ == "__main__":
    pred_sequences = [['O', 'S-LOC', 'O', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'O', 'E-ORG', 'O', 'B-PER', 'I-PER', 'E-PER']]
    gold_sequences = [['O', 'B-LOC', 'E-LOC', 'O', 'B-PER', 'E-PER'],
                    ['O', 'S-MISC', 'B-ORG', 'E-ORG', 'O', 'B-PER', 'E-PER', 'S-LOC']]
    test(pred_sequences, gold_sequences)