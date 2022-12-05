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
        final_start_idx = torch.argmax(start_prob).cpu()
        final_end_idx = torch.argmax(end_prob).cpu()
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

def align_index(input_ids, tokenizer, word_spans=None, document_start_index=None, sep_id=None, space_token=space_token):
    tokens = tokenizer.convert_ids_to_tokens(input_ids) 
    token_spans = []
    for ans, word_start, word_end in word_spans:
        assert word_start<=word_end
        word_index = 0
        find_start = False
        find_end = False
        for index in range(document_start_index+1, len(input_ids)):
            t = tokens[index]
            if input_ids[index]==sep_id or (t.startswith(space_token) and index!=document_start_index+1):
                word_index += 1
            if word_index == word_start and not find_start:
                token_start = index
                find_start = True
            if word_index == word_end + 1 and not find_end:
                token_end = index - 1
                find_end = True
                break
        # check 
        hstr = ''
        if find_start and find_end:
            for index in range(token_start, token_end + 1):
                hstr += tokens[index].strip('Â')
            hstr = hstr.replace('Ġ', ' ').strip()
            hstr = hstr.replace('▁', ' ').strip()
        '''
        if not (find_start and find_end) or hstr!=ans:
            print(tokens)
            print(ans)
            print(word_spans)
            print(token_start)
            print(token_end)
            print('start_token:', tokens[token_start])
            print('end_tokne:', tokens[token_end])
        '''
        #assert hstr==ans, f'{hstr}\t{ans}'
        if hstr==ans:
            token_spans.append([ans, token_start, token_end])
        else:
            token_spans.append([None, None, None])

    return token_spans
