import os
import argparse
import json
import numpy as np
import ast
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import ViltModel, ViltProcessor, ViltImageProcessor, ViltConfig
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

class WebQADataset(Dataset):

    def __init__(self, records, processor, image_processor, root):
        #self.questions = questions
        self.records = records
        self.processor = processor
        self.image_processor = image_processor
        self.root = root
        self.encoded_image_list = self.load_data()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # # get image + text
        # record = self.records[idx]
        # #questions = self.questions[idx]
        # image = Image.open(self.root+record['imagePath'])
        # text = record['text']

        # encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # encoding_image_processor = self.image_processor(image, do_resize=False, return_tensors="pt")
        # encoding.update(encoding_image_processor)

        # # remove batch dimension
        # for k,v in encoding.items():
        #   encoding[k] = v.squeeze()
        # # add labels
        # if 'patch_label' in record.keys():
        #     encoding["labels"] = record['patch_label'].float()
        return self.encoded_image_list[idx]

    def load_data(self):
        encoded_image_list = []
        for record in tqdm(self.records, desc='Processing images'):
            image = Image.open(self.root+record['imagePath'])
            text = record['text']

            encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
            encoding_image_processor = self.image_processor(image, do_resize=False, return_tensors="pt")
            encoding.update(encoding_image_processor)

            # remove batch dimension
            for k,v in encoding.items():
                encoding[k] = v.squeeze()
            # add labels
            if 'patch_label' in record.keys():
                encoding["labels"] = record['patch_label'].float()

            encoded_image_list.append(encoding)

        return encoded_image_list



class PatchSelectorModel(nn.Module):
    def __init__(self, number_patches):
        super(PatchSelectorModel, self).__init__()

        self.number_patches = number_patches
        # Instantiate ViltModel
        self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vilt_model.classifier = torch.nn.Identity()

        # config = self.vilt_model.config
        # config.hidden_dropout_prob = 0.2 
        # self.vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm", config=config)
        

        # Define the linear layer and sigmoid function
        # self.linear_layer = nn.Linear(self.vilt_model.config.hidden_size, 1)

        self.linear_layer  = torch.nn.Sequential(
          torch.nn.Linear(self.vilt_model.config.hidden_size, self.vilt_model.config.hidden_size//2),
          torch.nn.GELU(),
          torch.nn.Linear(self.vilt_model.config.hidden_size//2, self.vilt_model.config.hidden_size//4),
          torch.nn.GELU(),
          torch.nn.Linear(self.vilt_model.config.hidden_size//4, 49)
        )
        
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, labels):
        # Pass input through ViltModel
        vilt_output = self.vilt_model(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, pixel_values=pixel_values,
                                      pixel_mask=pixel_mask)

        # get image patches representations
        #linear_input = vilt_output.last_hidden_state[:, -self.number_patches:, :]
        linear_input = vilt_output.pooler_output


        # Pass through linear layer and sigmoid
        linear_output = self.linear_layer(linear_input)


        #sigmoid_output = self.sigmoid(linear_output)  #remove this

        return linear_output
    
    def load_model_weights(self, model_path):
        # Load weights from the specified path
        self.load_state_dict(torch.load(model_path))
    
    def get_weights_and_representation(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask):

        vilt_output = self.vilt_model(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, pixel_values=pixel_values,
                                      pixel_mask=pixel_mask)

        # get image patches representations
        # linear_input = vilt_output.last_hidden_state[:, -self.number_patches:, :]
        linear_input = vilt_output.pooler_output


        # Pass through linear layer and sigmoid
        linear_output = self.linear_layer(linear_input)
        sigmoid_output = self.sigmoid(linear_output)  

        return sigmoid_output, linear_input


def collate_fn(batch, image_processor):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]

    batch_for_training = False
    if 'labels' in batch[0].keys():
        labels = [item['labels'] for item in batch]
        batch_for_training = True

    # create padded pixel values and corresponding pixel mask
    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']

    if batch_for_training:
        batch['labels'] = torch.stack(labels)

    return batch

def accuracy(y_scores, y_true, threshold):
    return 100*np.sum((np.array(y_scores) > threshold) == np.array(y_true)) / len(y_true)


def plot_roc(fpr, tpr, roc_auc, out_filename):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(out_filename)


def read_and_prep_data(root, annotated_data_json, number_patches, processor, image_processor, for_training = True):
    
    f = open(annotated_data_json)
    records = json.load(f)

    if for_training:
        for record in records:
            labels = torch.zeros(number_patches, dtype=int)

            string_list = record['patches']
            int_list = ast.literal_eval(string_list)

            if len(int_list) > 0:
                labels[np.array(int_list)] = 1

            record['patch_label'] = labels


    if not os.path.exists(root+'/images_all_combined/public/'):
        print(root+'/images_all_combined/public/' + ' not found')        

    dataset = WebQADataset(records=records,
                            processor=processor,
                            image_processor=image_processor,
                            root=root+'/images_all_combined/public/')
    return dataset

def get_dataloader(root, data_file, number_patches, batch_size, shuffle = False, for_training = True):
    
    # read training annotated data in JSON file
    data = root+'/images_all_combined/'+data_file

    if not os.path.exists(data):
        print(data + 'not found.')
        return    

    # processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    # image_processor = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    image_processor = ViltImageProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    data_set = read_and_prep_data(root, data, number_patches, processor, image_processor, for_training)    

    def wrapper_collate_fn(batch):
        return collate_fn(batch, image_processor)

    dataloader = DataLoader(data_set, collate_fn=wrapper_collate_fn, batch_size=batch_size, shuffle=shuffle)    

    return dataloader



def train_patch_selector(root, training_data_file, validation_data_file, number_epochs, batch_size, learning_rate, positive_weight):

    TOTAL_NUMBER_PATCHES = 49

    if not os.path.exists(root):
        print('root provided does not exist')
        return

    # name of directory where model weights and figures will be saved
    output_folder = '/nepch_' + str(number_epochs) + '_bs_' + str(batch_size) + '_lr_' + str(learning_rate)

    if 'ablation' in training_data_file:
        ablation_size = training_data_file.split("_")[1].split(".")[0]
        output_folder = '/ablation/train_size' + ablation_size

    # create output directory if it already does not exist
    if not os.path.exists(root+output_folder):
        os.mkdir(root+'/'+output_folder)

    train_dataloader = get_dataloader(root, training_data_file, TOTAL_NUMBER_PATCHES, batch_size, True)
    val_dataloader = get_dataloader(root, validation_data_file, TOTAL_NUMBER_PATCHES, batch_size)
    
    model = PatchSelectorModel(TOTAL_NUMBER_PATCHES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    number_batches_train = len(train_dataloader)
    number_batches_val = len(val_dataloader)

    #criterion = nn.BCELoss()  # change loss to BCEWithLogits
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    #best_roc_auc = 0.0
    best_val_loss = np.inf
    for epoch in range(number_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(**batch)

            loss = criterion(outputs.squeeze(-1), batch['labels'])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= number_batches_train

        model.eval()
        val_loss = 0.0

        val_y_true = []
        val_y_scores = []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                loss = criterion(outputs.squeeze(-1), batch['labels'])                
                val_loss += loss.item()
                
                val_y_true.extend(list(batch['labels'].cpu().numpy().reshape(-1)))
                val_y_scores.extend(list(nn.Sigmoid()(outputs.squeeze(-1)).cpu().numpy().reshape(-1)))

        
        average_val_loss = val_loss / number_batches_val

        train_losses.append(epoch_loss)
        val_losses.append(average_val_loss)

        fpr, tpr, thresholds = roc_curve(val_y_true, val_y_scores)
        roc_auc = auc(fpr, tpr)

#        if roc_auc > best_roc_auc:
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss

            # Plot ROC curve
            plot_roc(fpr, tpr, roc_auc, root+output_folder+'/roc_plot.png')

            # save the model weights
            torch.save(model.state_dict(), root + output_folder + '/best_patch_selector_weights.pth')

            # save arrays to later help choose threshold
            np.savez(root + output_folder + '/roc_arrays.npz',
                     fpr=fpr, tpr=tpr, thresholds=thresholds,
                     y_true=np.array(val_y_true, dtype=int),
                     y_score=np.array(val_y_scores))

        acc25 = accuracy(val_y_scores, val_y_true, 0.25)
        acc50 = accuracy(val_y_scores, val_y_true, 0.5)
        acc75 = accuracy(val_y_scores, val_y_true, 0.75)

        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {average_val_loss:.4f}, ROC AUC: {roc_auc:.4f}, Acc25: {acc25:.2f}%, Acc50:{acc50:.2f}%, Acc75:{acc75:.2f}%")   
    

    # Plot loss function
    plt.clf()
    plt.plot(range(number_epochs), train_losses, label='Training Loss')
    plt.plot(range(number_epochs), val_losses, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')

    # Add legend
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(root+output_folder+'/training_validation_plot.png')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', dest='root', type=str, default='./', help='Project root directory')
    parser.add_argument('--train-data-file', dest="train_data_file", type=str, default='annotated_train_data_0.json', help='Name of json file with training annotated data.')
    parser.add_argument('--val-data-file', dest="val_data_file", type=str, default='annotated_val_data.json', help='Name of json file with validation annotated data.')
    parser.add_argument('--number-epochs', dest="number_epochs", type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', dest="batch_size", type=int, default = 8, help="Batch size")
    parser.add_argument('--lr', dest="lr", type=float, default = 5e-5, help='Learning rate')   
    parser.add_argument('--pos-weight', dest="pos_weight", type=float, default = 3.47, help='positive weight')  
    
    return parser.parse_args()


def get_weighted_representations(root, model_weights, data_file):

    TOTAL_NUMBER_PATCHES = 49

    # load model
    model = PatchSelectorModel(TOTAL_NUMBER_PATCHES)
    model.load_model_weights(root+model_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = get_dataloader(root, data_file, TOTAL_NUMBER_PATCHES, 8, False, False)

    model.eval()
    with torch.no_grad():
        representations = []
        for batch in tqdm(dataloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            w_b, r_b = model.get_weights_and_representation(**batch)
            w_b = w_b.squeeze(-1).cpu().numpy()
            r_b = r_b.cpu().numpy()

            for i in range(w_b.shape[0]):
                w_r = np.sum(w_b[i][:, np.newaxis]*r_b[i], axis=0)
                w_r = w_r.reshape((1,r_b.shape[-1]))
                representations.append(w_r)

    representations = np.vstack(representations)

    return representations

if __name__ == "__main__":

    args = parse_arguments()

    print(f"training data file: {args.train_data_file}, validation data file: {args.val_data_file}, number of epochs:{args.number_epochs}")
    print(f"batch size: {args.batch_size}, lr: {args.lr},pos weight: {args.pos_weight}")

    train_patch_selector(root= args.root,
                         training_data_file=args.train_data_file,
                         validation_data_file=args.val_data_file,
                         number_epochs=args.number_epochs,
                         batch_size=args.batch_size,
                         learning_rate=args.lr,
                         positive_weight=args.pos_weight)

    # r = get_weighted_representations(root = '/home/piocitos/docs/CMU/MultimodalML/finals/patch_selector', 
    #                                  model_weights = '/nepch_30_bs12_lr_5e-05/best_patch_selector_weights.pth', 
    #                                  data_file = 'sample_img_questions_unlabeled.json')
    

    # import pandas as pd

    # model = PatchSelectorModel(49)
    # root = '/home/piocitos/docs/CMU/MultimodalML/finals/patch_selector/nepch_30_bs12_lr_5e-05/'
    # model.load_model_weights(root+'best_patch_selector_weights.pth')

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)




    # root = '/home/piocitos/docs/CMU/MultimodalML/finals/patch_selector'
    # TOTAL_NUMBER_PATCHES = 49
    # batch_size = 8
    # data_file = 'annotated_val_data.json'
    # threshold_low = 0.1
    # threshold_up = 0.9
    # acceptable_conf = 0.8

    # dataloader = get_dataloader(root, data_file, TOTAL_NUMBER_PATCHES, batch_size)

    # prob_patches = []
    # with torch.no_grad():
    #     for batch in dataloader:
    #         batch = {k:v.to(device) for k,v in batch.items()}
    #         outputs = model(**batch)
    #         prob_patches_batch = model.sigmoid(outputs)
    #         prob_patches.append(prob_patches_batch.squeeze(-1).cpu().numpy())

    # prob_patches = np.vstack(prob_patches)

    # confidence_patch_label = np.logical_or(prob_patches > threshold_up, prob_patches < threshold_low)

    # confidence_patch_label = np.sum(confidence_patch_label, axis=1)/TOTAL_NUMBER_PATCHES

    # print(confidence_patch_label)





    # example on how to call this function
    # w, r = get_weights_and_representations(root = '/home/piocitos/docs/CMU/MultimodalML/final', 
    #                                        model_weights = '/nepch_10_bs_8_lr_5e-05/best_patch_selector_weights.pth',
    #                                        data_file = '/images_all_combined/all_img_questions_unlabeled.json')


