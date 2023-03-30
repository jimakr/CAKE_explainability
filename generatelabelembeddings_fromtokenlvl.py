import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from scipy.special import softmax
from optimus_code.dataset import Dataset
import heavenCashe
from keyphraseexpln import CAKE
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from xxhash import xxh3_64 as xhash
import os
import time
from optimus_code.modelwrapper import ModelWrap
from transformers import Trainer, TrainingArguments
import importlib


dataset_param = pd.read_excel('datasets.xlsx')
dataset_param = dataset_param.set_index('id')
old_parameters = pd.read_excel('parameters.xlsx')
parameters = old_parameters[old_parameters['F'].isnull()]
parameters = parameters.groupby(['Dataset', 'Distil', 'Embedding model'])
parameters = list(parameters)
if len(parameters) == 0:
    parameters = {'Dataset': 'Movies', 'Distil': False, 'Embedding model': 'Sbert', 'Keyphrase Number': 5, 'Threshold': 0.05, 'Ranking': 'cosine', 'Calibration': 'isotonic', 'Candidate Generation': 'PatternRank', 'AUPRC': np.nan, 'F': np.nan, 'FTP': np.nan, 'NZW': np.nan, 'time': np.nan, 'f1_fidelity': np.nan}
    parameters = {i: [j] for i, j in parameters.items()}
    parameters = pd.DataFrame.from_dict(parameters)
else:
    parameters = parameters[0][1]

dataset_param = dataset_param.loc[parameters.iloc[0]['Dataset']].to_dict()


dataname = dataset_param['dataname']
data_path = ''
save_path = f'./Results/{dataname}/'

existing_rationales = dataset_param['rationale']
task = dataset_param['task'] + '_label'
sentence_level = dataset_param['sentence']
distil = int(parameters.iloc[0]['Distil'])
labels = dataset_param['labels']
saverate = 100



if distil:
    distil = 'Distil'
else :
    distil = ''
if task=='multi_label':
    multi = 'Multilabel'
else:
    multi = ''

transformers = importlib.import_module("transformers")
used_tokenizer = getattr(transformers, f"{distil}BertTokenizerFast")

myTransformer = importlib.import_module("myTransformer")
used_model = getattr(myTransformer, f"{distil}BertFor{multi}SequenceClassification")

cased_string = '' if dataname=='Ethos'or dataname=='AIS' else 'un'
tokenizer = used_tokenizer.from_pretrained(f'{distil.lower()}bert-base-{cased_string}cased')

model_path = f'{distil.lower()}bert_{dataname.lower()}'
tokenizer.save_pretrained(model_path + '/')
model = used_model.from_pretrained(model_path, output_attentions=False, output_hidden_states=False)

training_arguments = TrainingArguments(evaluation_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
                                       log_level='critical', output_dir='./results', num_train_epochs=1,disable_tqdm=True ,
                                       per_device_train_batch_size=8, per_device_eval_batch_size=8,
                                       warmup_steps=200, weight_decay=0.01, logging_dir='./logs'
                                       )

trainer = Trainer(model=model, args=training_arguments)

model = ModelWrap(trainer, tokenizer, task, labels)

hoc = Dataset(path='')
if dataname == 'HOC':
    x, y, label_names, rationales = hoc.load_hoc()
elif dataname == 'HX':
    x, y, label_names, rationales = hoc.load_hatexplain(tokenizer)
elif dataname == 'Movies' and not sentence_level:
    x, y, label_names, rationales = hoc.load_movies(level='token')
elif dataname == 'Movies' and sentence_level:
    x, y, label_names, rationales = hoc.load_movies(level='sentence')
elif dataname == 'Hummingbird':
    x, y, label_names, rationales = hoc.load_hummingbird()
elif dataname == 'Ethos':
    x, y, label_names = hoc.load_ethos()
    label_names = label_names[1:]  # Ethos
elif dataname == 'ESNLI':
    dataset, label_names = hoc.load_esnli()
    train_texts, test_texts, test_rationales, validation_texts, train_labels, test_labels, validation_labels = dataset
elif dataname == 'AIS':
    x, y, label_names = hoc.load_AIS()
    label_names = ['class a', 'class b']

if dataname != 'ESNLI':
    indices = np.arange(len(y))
    train_texts, test_texts, train_labels, test_labels, _, test_indexes = train_test_split(x, list(y), indices,
                                                                                           test_size=.2,
                                                                                           random_state=42)
    if existing_rationales:
        test_rationales = [rationales[x] for x in test_indexes]

    size = (0.1 * len(y)) / len(train_labels)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(list(train_texts), train_labels,
                                                                                      test_size=size, random_state=42)


if dataname=='Movies':
    train_texts.append(test_texts[84])
    train_labels.append(test_labels[84])
    train_texts.append(test_texts[72])
    train_labels.append(test_labels[72])
    test_texts.pop(84)
    test_labels.pop(84)
    test_rationales.pop(84)
    test_texts.pop(72)
    test_labels.pop(72)
    test_rationales.pop(72)
    test_texts.pop(63)
    test_labels.pop(63)
    test_rationales.pop(63)

if existing_rationales and dataname == 'HOC':
    test_label_rationales = []
    for test_rational in test_rationales:
        label_rationales = []
        for label in range(labels):
            label_rationales.append([])
        for sentence in test_rational:
            for label in range(labels):
                if label_names[label] in sentence:
                    label_rationales[label].append(1)
                else:
                    label_rationales[label].append(0)
        test_label_rationales.append(label_rationales)
    final_rationales = test_label_rationales

elif existing_rationales and dataname == 'HX':
    test_test_rationales = []
    for test_rational in test_rationales:
        test_test_rationales.append([0, test_rational])
    final_rationales = test_test_rationales
elif existing_rationales and dataname == 'ESNLI':
    for i in range(len(test_rationales)):
        if (test_rationales[i][0] == []):
            test_rationales[i][0] = [0] * len(test_rationales[i][1])
            test_rationales[i][1] = list(test_rationales[i][1])
        else:
            test_rationales[i][1] = [0] * len(test_rationales[i][0])
            test_rationales[i][0] = list(test_rationales[i][0])
    test_test_rationales = test_rationales

    new_rationale = []
    for i in range(2000):
        rationale = []
        test_t = test_texts[i].split(' ')
        for j in range(2):
            label_rational = []
            for k in range(len(test_t)):
                for r in tokenizer.tokenize(test_t[k]):
                    # if r == '.':
                    #    print(r)
                    # print(r)
                    rationall = 1 if test_test_rationales[i][j][k] > 0 else 0
                    label_rational.append(rationall)
            rationale.append(label_rational)
        new_rationale.append(rationale)
    final_rationales = new_rationale

elif existing_rationales and dataname == 'Movies' and not sentence_level:
    test_test_rationales = []
    for i in range(len(test_rationales)):
        if (test_labels[i] == 1):
            test_test_rationales.append([[0] * len(test_rationales[i]), test_rationales[i]])
        else:
            test_test_rationales.append([test_rationales[i], [0] * len(test_rationales[i])])

    new_rationale = []
    for i in range(len(test_test_rationales)):
        rationale = []
        test_t = test_texts[i].split(' ')
        for j in range(2):
            label_rational = []
            for k in range(len(test_t)):
                for r in tokenizer.tokenize(test_t[k]):
                    # if r == '.':
                    #    print(r)
                    # print(r)
                    rationall = 1 if test_test_rationales[i][j][k] > 0 else 0
                    label_rational.append(rationall)
            rationale.append(label_rational)
        new_rationale.append(rationale)
    final_rationales = new_rationale

elif existing_rationales and dataname == 'Movies' and sentence_level:
    test_test_rationales = []
    for i in range(len(test_rationales)):
        if (test_labels[i] == 1):
            test_test_rationales.append([[0] * len(test_rationales[i][:-1]), list(test_rationales[i][:-1])])
        else:
            test_test_rationales.append([list(test_rationales[i][:-1]), [0] * len(test_rationales[i][:-1])])
    final_rationales = test_test_rationales

elif existing_rationales and dataname == 'Hummingbird':
    new_rationale = []
    for i in range(len(test_rationales)):
        rationale = []
        test_t = test_texts[i].split(' ')
        for j in range(6):
            label_rational = []
            for k in range(len(test_t)):
                for r in tokenizer.tokenize(test_t[k]):
                    rationall = 1 if test_rationales[i][j][k] > 0 else 0
                    label_rational.append(rationall)
            rationale.append(label_rational)
        new_rationale.append(rationale)

    final_rationales = new_rationale
else:
    final_rationales = test_texts

if parameters.iloc[0]['Embedding model'] == 'Trained':
    kek = CAKE(label_names, 15, model_path=model_path)

else:
    kek = CAKE(label_names, 15)


kek.train_vectorizer(train_texts)
kek.document_and_word_embed(train_texts, unique=f'{distil}Bert')



predictor = heavenCashe.instansheCashe(dataname, f'{distil}Bert', saverate)(model.single_predict)

train_predictions = []
for train_text in train_texts:
    outputs = predictor(train_text.encode('ascii', 'ignore').decode())
    train_predictions.append(outputs)

predictions = []
for test_text in test_texts:
    outputs = predictor(test_text)
    predictions.append(outputs)

y_model = np.array(train_labels)

print(predictions[7])

if task == 'multi_label':
    import tensorflow as tf

    a = tf.constant(predictions, dtype=tf.float32)
    b = tf.keras.activations.sigmoid(a)
    predictions = b.numpy()

    a = tf.constant(train_predictions, dtype=tf.float32)
    b = tf.keras.activations.sigmoid(a)
    train_predictions = b.numpy()

    pred_labels = []
    for prediction in predictions:
        pred_labels.append([1 if i >= 0.5 else 0 for i in prediction])

    train_pred_labels = []
    for prediction in train_predictions:
        train_pred_labels.append([1 if i >= 0.5 else 0 for i in prediction])
    multi = True

elif task == 'single_label':
    pred_labels = []
    for prediction in predictions:
        pred_labels.append(np.argmax(softmax(prediction)))

    train_pred_labels = []
    for prediction in train_predictions:
        train_pred_labels.append(np.argmax(softmax(prediction)))

    multi = False


identifier = f'{dataname} {str(distil)}'
if not os.path.exists(f'./token_generated/{xhash(identifier).intdigest()}'):
    ts = time.time()
    emb = TransformerWordEmbeddings(model_path, layers='-1', layer_mean=False, use_context=True)
    print('starting')

    print(f'length {len(test_texts)}')
    label_embeddings = [[] for i in label_names]
    for train_text in tqdm(train_texts):
        for i, label in enumerate(label_names):

            sent = (Sentence(f'Label: {label} describes document: {train_text}'))
            embeddings = emb.embed(sent)

            label_emb = []
            for token in embeddings[0].tokens[2:2+len(label.split(' '))]:
                label_emb.append(np.array(token.embedding.cpu()))

            label_embeddings[i].append(np.mean(label_emb, axis=0))

    label_embeddings = np.array(label_embeddings)
    print(label_embeddings.shape)
    print(label_embeddings[0].shape)
    label_embeddings = np.mean(label_embeddings, axis=1)
    # print('embedding')
    identifier = f'{dataname} {distil}'
    with open(f'token_generated/{xhash(identifier).intdigest()}', 'wb') as f:
        pickle.dump(label_embeddings, f)
    ts = time.time() - ts
    print(f'time {ts}')
else:
    identifier = f'{dataname} {str(distil)}'
    print(f'already calculated these values with {identifier}')







