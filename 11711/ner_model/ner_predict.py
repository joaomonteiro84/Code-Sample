import os
import torch
import argparse
from transformers import AutoTokenizer   
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from torch.utils.data import DataLoader
from helper_functions import get_tag_dict, get_dataset, check_paths_predict 
from helper_functions import tokenize_and_align_labels, get_prediction_words
from helper_functions import write_predictions_conll_file, write_predictions_csv_file


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', dest='root', type=str, default='./', help='Project root directory')
    parser.add_argument('--ner-model-name', dest="model_name", type=str, default='bert-finetuned-ner', help='Name of the ner model to use for predict ner tags.')
    parser.add_argument('--test-data', dest="test_file", type=str, help='Name of .txt file with the test data.')
    parser.add_argument('--batch-size', dest="batch_size", type=int, default = 8)
    parser.add_argument('--max-number-tokens', dest="max_number_tokens", type=int, default = 350)
    parser.add_argument('--ner-pred-save-file', dest="ner_pred_save_file", type=str, help = "Name of file to save ner predictions")
    
    return parser.parse_args()

def predict_and_save(root = './',
                     model_name = 'bert-finetuned-ner',
                     test_file = 'bert.txt',
                     batch_size = 8,
                     max_number_tokens = 350,
                     device='cuda',
                     ner_pred_save_file ='bert_pred.connl'):

  if not os.path.exists(root):
    print(f"Cannot access root at '{root}'.")
    return False

  model_path = root + model_name
  test_file = root + test_file
  ner_pred_save_file = root + ner_pred_save_file

  #check paths
  all_paths_good = check_paths_predict(model_path, test_file, ner_pred_save_file)

  if not all_paths_good:
    return False

  # load model
  model = AutoModelForTokenClassification.from_pretrained(model_path)

  # Move the model to CUDA
  model = model.to(device)

  # load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  # load data from file
  test_dataset = get_dataset(input_file = test_file, max_number_tokens=max_number_tokens)

  # data collator to build paddings accordinly to the backbone model
  data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

  tokenized_test_set = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=test_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

  test_data = DataLoader(tokenized_test_set, batch_size=batch_size, 
                        shuffle=False, collate_fn=data_collator)

  # create a dictionary from ner tags (string) to id (an integer)
  label2id = get_tag_dict()
  label_names = list(label2id.keys())

  # create a dictionary from id (an integer) to the ner tags (string)
  id2label = {i: label for i, label in enumerate(label_names)}

  # get predictions
  ner_preds = []
  for batch_number, batch in enumerate(test_data):
    batch_device = batch.to(device)
    model_output = model(**batch_device)

    # this includes predictions for subwords and special tokens such as CLS and SEP
    sub_words_predictions = torch.argmax(model_output.logits, dim=2) 

    # select only predictions for the words. if a word is 
    # divided into subwords, only keep the prediction for the first subword
    for i in range(len(sub_words_predictions)):

      original_tokens = test_dataset["tokens"][batch_number*batch_size + i]

      np = get_prediction_words(sw_pred = sub_words_predictions[i].cpu(), 
                                labels = batch['labels'][i].cpu(), 
                                original_tokens = original_tokens,
                                id2label =  id2label)
      
      ner_preds.append(np)



  _, file_extension = os.path.splitext(ner_pred_save_file)  

  if file_extension == '.csv':
    write_predictions_csv_file(ner_preds, ner_pred_save_file, test_dataset['sentences_id'])
  else:
    write_predictions_conll_file(ner_preds, ner_pred_save_file, test_dataset['sentences_id'])

  return True

if __name__ == '__main__':
    #get arguments
    args = parse_arguments()

    predictions_done = predict_and_save(root = args.root,
                                        model_name = args.model_name,
                                        test_file = args.test_file,
                                        batch_size = args.batch_size,
                                        max_number_tokens=args.max_number_tokens,
                                        device='cuda',
                                        ner_pred_save_file = args.ner_pred_save_file)
    
    if not predictions_done:
       print('Predictions failed')
    else:
       print('Done')

