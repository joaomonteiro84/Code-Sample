import os
import numpy as np
import csv
from io import StringIO
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# util functions
def create_directory(directory_path):
  # Check if the directory already exists
  if not os.path.exists(directory_path):
      # Create the directory if it doesn't exist
      os.makedirs(directory_path)
      print(f"Directory '{directory_path}' created.")
  else:
      print(f"Directory '{directory_path}' already exists.")


def get_tag_dict():
  # create a corresponding integer for each tag
  ner_tag_int = {}
  ner_tag_int['O'] = 0

  tags = ['MethodName', 'HyperparameterName', 'HyperparameterValue',
          'MetricName', 'MetricValue', 'TaskName', 'DatasetName']

  counter = 1
  for t in tags:
    ner_tag_int['B-'+t] = counter
    ner_tag_int['I-'+t] = counter+1
    counter +=2

  return ner_tag_int

# function copied from Hugging Face. source: 
# https://huggingface.co/docs/transformers/tasks/token_classification
def tokenize_and_align_labels(examples, tokenizer):
  tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

  labels = []
  for i, label in enumerate(examples[f"ner_tags"]):
      word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
      previous_word_idx = None
      label_ids = []
      for word_idx in word_ids:  # Set the special tokens to -100.
          if word_idx is None:
              label_ids.append(-100)
          elif word_idx != previous_word_idx:  # Only label the first token of a given word.
              label_ids.append(label[word_idx])
          else:
              label_ids.append(-100)
          previous_word_idx = word_idx
      labels.append(label_ids)

  tokenized_inputs["labels"] = labels
  
  return tokenized_inputs

# function copied from from Hugging Face. 
# source: https://huggingface.co/docs/transformers/tasks/token_classification
def compute_metrics(p, label_names, metric):
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)

  #label_names = list(get_tag_dict().keys())

  true_predictions = [
      [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]
  true_labels = [
      [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]

  results = metric.compute(predictions=true_predictions, references=true_labels)
  return {
      "precision": results["overall_precision"],
      "recall": results["overall_recall"],
      "f1": results["overall_f1"],
      "accuracy": results["overall_accuracy"],
  }

def prep_data(lines, max_number_tokens = 300):

  ner_tag_dict = get_tag_dict()

  sentences_id = []
  sentences = []
  tags = []
  tags_int = []

  running_sentence = []
  running_tags = []
  running_tags_int = []

  number_tokens = 0
  sid = 0
  for l in lines:
    if l == "\n":
      sentences.append(running_sentence)
      tags.append(running_tags)
      tags_int.append(running_tags_int)
      running_sentence = []
      running_tags = []
      running_tags_int = []
      number_tokens = 0
      sentences_id.append(sid)
      sid += 1
    else:
      word, tag = l.split()
      running_sentence.append(word)
      running_tags.append(tag)
      running_tags_int.append(ner_tag_dict[tag])
      number_tokens += 1
      if number_tokens == max_number_tokens:
        sentences_id.append(sid)
        sentences.append(running_sentence)
        tags.append(running_tags)
        tags_int.append(running_tags_int)
        running_sentence = []
        running_tags = []
        running_tags_int = []
        number_tokens = 0

  if len(running_sentence) > 0:
    sentences_id.append(sid)
    sentences.append(running_sentence)
    tags.append(running_tags)
    tags_int.append(running_tags_int)

  return sentences_id, sentences, tags, tags_int


def prep_data_no_tags(lines, max_number_tokens = 300):

  sentences_id = []
  sentences = []
  tags_int = []
  
  running_sentence = []
  running_tags_int = []

  number_tokens = 0
  sid = 0
  iter=0
  for l in lines:
    iter+=1
    csv_reader = csv.reader(StringIO(l))
    _, word, _ = next(csv_reader)

    if word == "":
      sentences.append(running_sentence)   
      tags_int.append(running_tags_int)   
      running_sentence = []
      running_tags_int = []
      number_tokens = 0
      sentences_id.append(sid)
      sid += 1
    else:      
      running_sentence.append(word)
      running_tags_int.append(0)
      number_tokens += 1
      if number_tokens == max_number_tokens:
        sentences_id.append(sid)
        sentences.append(running_sentence)      
        tags_int.append(running_tags_int)  
        running_sentence = []
        running_tags_int = []
        number_tokens = 0

  if len(running_sentence) > 0:
    sentences.append(running_sentence)   
    tags_int.append(running_tags_int)
    sentences_id.append(sid)

  return sentences_id, sentences, tags_int


def load_dataset_with_label(input_file, max_number_tokens):

  with open(input_file, 'r') as file:
    lines = file.readlines()

  sentences_id, sentences, _, tags_int = prep_data(lines, max_number_tokens)

  return sentences_id, sentences, tags_int


def get_train_val_dataset(input_train_file, input_val_file, max_number_tokens = 350):

  _, train_s, train_tag = load_dataset_with_label(input_train_file, max_number_tokens)
  _, val_s, val_tag = load_dataset_with_label(input_val_file, max_number_tokens)

  dataset = DatasetDict({'train': Dataset.from_dict({"tokens": train_s, "ner_tags": train_tag}),
                         'validation': Dataset.from_dict({"tokens": val_s, "ner_tags": val_tag})})

  return dataset


# input file is a .txt file with one paragraph per line
# or a .csv file with 3 columns (id, input, target) where the first
# two lines are headeres to be ignored
def get_dataset(input_file, max_number_tokens = 350):
  with open(input_file, 'r') as file:
    lines = file.readlines()
  
  _, file_extension = os.path.splitext(input_file)

  if file_extension == '.csv':
    sentences_id, sentences, tags_int = prep_data_no_tags(lines[2:], max_number_tokens)
  else:
    new_lines = []
    for l in lines:
      new_lines += [token + ' O\n' for token in l.split()] + ['\n']

    sentences_id, sentences, _, tags_int = prep_data(new_lines, max_number_tokens)

  dataset = Dataset.from_dict({"sentences_id":sentences_id, "tokens": sentences, "ner_tags": tags_int})

  return dataset

def get_prediction_words(sw_pred, labels, original_tokens, id2label):
  predict_words = [p.item() for (p, b_l) in zip(sw_pred, labels) if b_l != -100]
  return [(t, id2label[p]) for (t, p) in zip(original_tokens, predict_words)]


def write_predictions_conll_file(predictions, file_path, sentences_id):
  n_pred_subsent = len(predictions)

  with open(file_path, 'w', encoding='utf-8') as file:
    for i in range(n_pred_subsent-1):
      p = predictions[i]
      for (word, tag_pred) in p:
        line = word+" "+tag_pred+"\n"
        file.write(line)

      if sentences_id[i] != sentences_id[i+1]:
        file.write('\n')

    p = predictions[i+1]
    for (word, tag_pred) in p:
      line = word+" "+tag_pred+"\n"
      file.write(line)

def write_predictions_csv_file(predictions, file_path, sentences_id):
  n_pred_subsent = len(predictions)

  with open(file_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['id', 'target'])
    csv_writer.writerow([1, 'X'])
    id = 2

    for i in range(n_pred_subsent-1):
      p = predictions[i]
      for (_, tag_pred) in p:
        csv_writer.writerow([id, tag_pred])
        id += 1

      if sentences_id[i] != sentences_id[i+1]:
        csv_writer.writerow([id, 'X'])
        id += 1

    p = predictions[i+1]
    for (_, tag_pred) in p:
      csv_writer.writerow([id, tag_pred])
      id += 1

def check_paths_predict(model_path, test_file, ner_pred_save_file):
  if not os.path.exists(model_path):
    print(f"Directory '{model_path}' not found.")
    return False

  if not os.path.exists(test_file):
    print(f"Data file '{test_file}' not found.")
    return False

  save_path_pieces = ner_pred_save_file.split('/')
  file_name = save_path_pieces[-1]
  save_path = "/".join(save_path_pieces[:-1])
  if not os.path.exists(save_path):
    print(f"Cannot save file '{file_name}' at {save_path} since this path was not found.")
    return False

  return True