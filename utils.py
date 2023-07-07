import torch
import torch.nn.functional as F


# ---------------------------------------------------------
# functions
def compare(pred_start, pred_end, gold_start, gold_end):
    if pred_start >= pred_end:
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    elif pred_end <= gold_start or pred_start >= gold_end:
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    elif gold_end == gold_start:
        overlap_start = 0
        overlap_end = 0
        Max = 0
        Min = 0
        no_overlap = True
    else:
        no_overlap = False
        if pred_start <= gold_start:
            Min = pred_start
            overlap_start = gold_start
        else:
            Min = gold_start
            overlap_start = pred_start

        if pred_end <= gold_end:
            Max = gold_end
            overlap_end = pred_end
        else:
            Max = pred_end
            overlap_end = gold_end

    return overlap_start, overlap_end, Min, Max, no_overlap

def _get_best_indexes(probs, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def post_process_prediction(start_prob, end_prob, n_best_size=10, max_answer_length=500, weight=0.5):
    prelim_predictions = []
    start_prob = start_prob.squeeze()
    end_prob = end_prob.squeeze()

    start_indexes = _get_best_indexes(start_prob, n_best_size)
    end_indexes = _get_best_indexes(end_prob, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant

    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            predict = {
                        'start_prob': start_prob[start_index],
                        'end_prob': end_prob[end_index],
                        'start_idx': start_index,
                        'end_idx': end_index,
                      }
            prelim_predictions.append(predict)

    prelim_predictions = sorted(prelim_predictions,
                                key=lambda x: ((1-weight)*x['start_prob'] + weight*x['end_prob']),
                                reverse=True)
    if len(prelim_predictions) > 0:
        final_start_idx = prelim_predictions[0]['start_idx']
        final_end_idx = prelim_predictions[0]['end_idx']
    else:
        final_start_idx = torch.argmax(start_prob).cpu().item()
        final_end_idx = torch.argmax(end_prob).cpu().item()
    return final_start_idx, final_end_idx

def F1EM_score(pred_start, pred_end, gold_start, gold_end):
    overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)
    if no_overlap:
        if pred_start == gold_start and pred_end == gold_end:
            F1 = 1.0
            EM = 1
        else:
            F1 = 0.0
            EM = 0
    else:
        Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
        Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        if F1==1.0:
            EM = 1
        else:
            EM = 0

    return F1,  EM

def align_index(tokenized_input, word_spans, tokenizer):
    # note: answer token_ids will be input_ids[token_start:token_end+1]
    word_ids = tokenized_input.word_ids()
    # get document start index (sep index + 1)
    document_start_index = 1
    while word_ids[document_start_index]!=None:
        document_start_index += 1
    document_start_index += 1
    # get token_start and token_end
    token_spans = []
    for ans, word_start, word_end, start_time, end_time in word_spans: 
        # get token_start
        try:
            token_start = document_start_index
            while word_ids[token_start]!=word_start:
                token_start += 1
            token_end = len(word_ids)-1
            while word_ids[token_end]!=word_end:
                token_end -= 1
        except:
            import pdb;pdb.set_trace()

        # check 
        '''
        hstr = ''
        tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0]) 
        for index in range(token_start, token_end + 1):
            hstr += tokens[index]
            #hstr += tokens[index].strip('Â')
        hstr = hstr.replace(tokens[token_start][0], ' ').strip()
        #hstr = hstr.replace('Ġ', ' ').strip()
        #hstr = hstr.replace('▁', ' ').strip()
        if hstr!=ans:
            print('token_start', token_start)
            print('token_end', token_end)
            print('found ans:', tokens[token_start:token_end+1])
            print('hstr:', hstr)
            print('ans:', ans)
            import pdb;pdb.set_trace()
        #assert hstr==ans, f'{hstr}\t{ans}'
        '''
        token_spans.append([ans, token_start, token_end, start_time, end_time])
        #if hstr==ans:
        #    token_spans.append([ans, token_start, token_end, start_time, end_time])
        #else:
        #    token_spans.append([None, None, None, None, None])
    return token_spans

def align_word_index(word_ids, token_span):
    '''
    convert token start/end index to word start/end index
    '''
    # get word_ids for document only (remove question and the second [SEP])
    document_start_index = 1
    while word_ids[document_start_index]!=None:
        document_start_index += 1
    document_start_index += 1
    word_ids = word_ids[document_start_index:]
    token_start_index, token_end_index = token_span
    token_end_index = min(token_end_index, len(word_ids)-2)
    word_start_index, word_end_index = word_ids[token_start_index], word_ids[token_end_index]
    return word_start_index, word_end_index
    
'''
def align_word_index(input_ids, tokenizer, pred_span=None, document_start_index=None, sep_id=None, space_token=None):
    tokens = tokenizer.convert_ids_to_tokens(input_ids) 
    word_spans = []
    pred_tokens = []
    token_start, token_end = pred_span
    if token_start>token_end:
        return [0, 0, []]
    assert token_start<=token_end
    word_index = 0
    find_start = False
    find_end = False
    #for index in range(document_start_index+1, len(input_ids)):
    for index in range(len(input_ids)):
        t = tokens[index]
        if find_start and not find_end:
            pred_tokens.append([index, t])
        new_word = False
        #if input_ids[index]==sep_id or (t.startswith(space_token) and index!=document_start_index+1):
        if input_ids[index]==sep_id or (t.startswith(space_token) and index!=0):
            word_index += 1
            new_word = True
        if index == token_start and not find_start:
            word_start = word_index
            find_start = True
            pred_tokens.append([index, t])
        if index == token_end + 1 and not find_end:
            if new_word:
                word_end = word_index - 1
                pred_tokens = pred_tokens[:-1]
            else:
                word_end = word_index
            find_end = True
            break
    if not (find_start and find_end):
        import pdb;pdb.set_trace()
    assert find_start and find_end
    return [word_start, word_end, pred_tokens]
'''

def get_document_start_index(input_ids, sep_id):
    document_start_index = 0
    while input_ids['input_ids'][0][document_start_index]!=sep_id:
        document_start_index += 1
    return document_start_index

def Frame_F1_scores(pred_starts, pred_ends, gold_starts, gold_ends):
    F1s = []
    for pred_start, pred_end, gold_start, gold_end in zip(pred_starts, pred_ends, gold_starts, gold_ends):
        overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)
        if no_overlap:
            if pred_start == gold_start and pred_end == gold_end:
                F1 = 1
            else:
                F1 = 0
        else:
            Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
            Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
            F1 = float(2 * Precision * Recall / (Precision + Recall))
        F1s.append(F1)
    return F1s

def Frame_F1_score(pred_start, pred_end, gold_start, gold_end):
    overlap_start, overlap_end, Min, Max, no_overlap = compare(pred_start, pred_end, gold_start, gold_end)
    if no_overlap:
        if pred_start == gold_start and pred_end == gold_end:
            F1 = 1
        else:
            F1 = 0
    else:
        Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
        Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
        F1 = 2 * Precision * Recall / (Precision + Recall)
    return F1

def approximate_span_time(word2time, word_span):
    start_word_index, end_word_index = word_span
    # get appro_start_time
    appro_start_word_index = start_word_index
    while appro_start_word_index>=0 and len(word2time[appro_start_word_index])!=4:
        appro_start_word_index -= 1
    if appro_start_word_index >= 0:
        appro_start_time = float(word2time[appro_start_word_index][2])
    else:
        appro_start_time = 0.0
    # get appro_end_time
    appro_end_word_index = end_word_index
    while appro_end_word_index < len(word2time) and len(word2time[appro_end_word_index])!=4:
        appro_end_word_index += 1
    if appro_end_word_index < len(word2time):
        appro_end_time = float(word2time[appro_end_word_index][3])
    else:
        index = len(word2time)-1
        while len(word2time[index])!=4:
            index -= 1
        appro_end_time = max(appro_start_time, float(word2time[index][3]))
    
    return appro_start_time, appro_end_time
