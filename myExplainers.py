import torch
from scipy.special import softmax
import numpy as np
from optimus_code.myModel import MyDataset
import tensorflow as tf
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import utility
from keyphraseexpln import CAKE
from transformers_interpret import SequenceClassificationExplainer

class MyExplainer:
	""" MyExplainer class contains the implementation of multiple interpretation techniques including LIME, IG and Attention (but not Optimus)"""

	def __init__(self, label_names, model, sentence_level=False, split_token='.', layers=12, heads=12):
		"""Init function
		Args:
			label_names: The label names of the dataset
			model: The transformer model (using our class MyModel)
			sentence_level: Boolean for sentence level interpretations (True) or token level ones (False)
			split_token: In case of sentence level interpretations the token on which the sentences will be split
			layers: The number of the layers (integer)
			heads: The number of the heads (integer)
		Attributes:
			tokenizer: The model specific tokenizer -> initiated through the __load_model__ function
			max_sequence_len: The max number of tokens per sequence (e.g. 512 in Bert/DistilBert Base)
			lime_explainer: The LIME explainer
			neighbours: The numbers of neighbours used in LIME
			ig_explainer: The IG explainer
			config: The attention configuration setup
			save_states: A dictionary for saving calculated configurations
		"""
		self.layers = layers
		self.heads=heads
		self.label_names = label_names
		self.tokenizer = model.tokenizer
		self.sentence_level = sentence_level
		self.split_token = split_token
		self.model = model
		self.max_sequence_len = self.tokenizer.max_len_single_sentence
		self.config = None
		self.save_states = {}
		self.neighbours = 2000

	def initiate_ig(self, ig):
		self.ig_explainer = ig

	def ig(self, instance, prediction, tokens, mask, attention, hidden_states):
		""" This function represents the IG explainer. From the arguments it uses only the tokens
		Args:
			tokens: The tokenized instance
		Return:
			interpretations: It returns the extracted interpretations per label
		"""
		interpretations = []
		for label in range(len(self.label_names)):
			explanations = [explanation[1] for explanation in self.ig_explainer(instance, index=label, internal_batch_size=10, n_steps=30)[1:-1]]
			interpretations.append(explanations)
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations

	def fix_instance(self, instance):
		""" This function will restore a tokenized instance to each original state. For example the tokens ["I", "am", "very", "smart", "##eous"] will become "I am very smarteous"
		Args:
			instance: The tokenized instance to be fixed
		"""
		new_sentence = ''
		temp_split = instance.split()
		for i in range(0,len(temp_split)):
			if "##" not in temp_split[i]:
				new_sentence = new_sentence + ' ' + temp_split[i]
			else:
				new_sentence = new_sentence + temp_split[i][2:]
		return new_sentence[1:]

	def convert_to_sentence(self, tokens, interpretations):
		""" This function converts an interpretation from token level to sentence level using the split_token defined in the initialization.
		Args:
			tokens: The tokenized instance
			interpretations: the interpretations for all the labels. Each interpretation has as many importance scores as the number of tokens. 
		Return:
			[label_sentences, label_interpretations]: Returns a list which contains a) label_sentences (the identified sentences) and b) label_interpretations (the interpretations per label for the identified sentences)
		"""
		if self.split_token != '.':
			abstract = ' '.join(tokens)+' '
			abstract = abstract.replace(' . ',' ‡ ')
			abstract = abstract.replace('.',' ‡ ')
			tokens = abstract.split()
		label_interpretations = []
		label_sentences = []
		for label in range(len(self.label_names)):
			sentences = []
			sentences_weights = []
			sentence = []
			sentence_weight = []
			flagoto= False
			for weight,token in zip(interpretations[label],tokens[1:-1]):
				if token == '‡' and flagoto:
					sentences.append(' ')
					sentences_weights.append(0.0)
					sentence = []
					sentence_weight = []
				else:
					if self.split_token != '.' and token == '‡':
						sentence.append('.')
						flagoto = True
					else:
						sentence.append(token)
						flagoto = False
					sentence_weight.append(weight)
					if token == self.split_token:
						sentences.append(self.fix_instance(' '.join(sentence)))
						sentences_weights.append(np.array(sentence_weight).mean()) #mean/max/sum
						sentence = []
						sentence_weight = []
				label_sentences.append(sentences)
			label_interpretations.append(sentences_weights)
		return [label_sentences, label_interpretations]

	def cake_explainer(self, instance, prediction, tokens, mask, attention, hidden_states):
		interpretations = []
		instance_ids = self.tokenizer(instance, truncation=True, padding=True)['input_ids']
		# return keyphrases and weights for every label
		keyphrases, weights = self.keyphraser.single_doc_keyword_with_knn(instance)

		for label in range(len(self.label_names)):
			keyphrases_ids = self.tokenizer(keyphrases[label], truncation=True)['input_ids']
			# mach back to text using id's
			intepretetion, matches = utility.keyphrase_ids_to_document_weights(keyphrases_ids, instance_ids, weights[label])
			interpretations.append(intepretetion[1:-1])
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations

	def dummy(self, instance, prediction, tokens, mask, attention, hidden_states):
		interpretations = []
		instance_ids = self.tokenizer(instance, truncation=True, padding=True)['input_ids']

		for label in range(len(self.label_names)):

			interpretations.append(np.zeros_like(instance_ids)[1:-1])
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations

	def initialize_keyphraser(self, keyphraser : CAKE):
		self.keyphraser : CAKE = keyphraser
		return


	def my_attention(self, instance, prediction, tokens, mask, attention_i, hidden_states):
		""" This function represents the lime explainer. From the arguments it uses only the tokens and attention_i
		Args:
			tokens: The tokenized instance
			attention_i: The attention matrices as calculated for the examined instance
		Return:
			interpretations: It returns the extracted interpretations per label
		"""
		layers = self.config[0] # Mean, Multi, Sum, First, Last
		heads = self.config[1] # Mean, Sum, First, Last
		matrix = self.config[2] # From, To, MeanColumns, MeanRows, MaxColumns, MaxRows
		selection = self.config[3] #True: select layers per head, False: do not
		
		attention = attention_i.copy()
		if not selection:
			
			if heads == 'Mean':
				if heads not in self.save_states:
					self.save_states[heads] = attention.mean(axis=1)
					attention = self.save_states[heads]
				else:
					attention = self.save_states[heads]
			elif heads == 'Sum':
				if heads not in self.save_states:
					self.save_states[heads] = attention.sum(axis=1)
					attention = self.save_states[heads]
				else:
					attention = self.save_states[heads]
			elif type(heads) == type(1):
				attention = attention[:,heads,:,:]
				
			if layers == 'Mean':
				attention = attention.mean(axis=0)
			elif layers == 'Sum':
				attention = attention.sum(axis=0)
			elif layers == 'Multi':
				joint_attention = attention[0]
				for i in range(1, len(attention)):
					joint_attention = np.matmul(attention[i],joint_attention)
				attention = joint_attention
			elif type(layers) == type(1):
				attention = attention[layers]

			if matrix == 'From':
				attention = attention[0]
			elif matrix == 'To':
				attention = attention[:,0]
			elif matrix == 'MeanColumns':        
				attention = attention.mean(axis=0)
			elif matrix == 'MeanRows':
				attention = attention.mean(axis=1)
			elif matrix == 'MaxColumns':
				attention = attention.max(axis=0)
			elif matrix == 'MaxRows':
				attention = attention.max(axis=1)
		else:
			importance_attention_matrices = []
			for i in range(self.layers): #TO VHANGE IN THE FUTURE?
				att_heads = []
				for j in range(self.heads): #TO VHANGE IN THE FUTURE?
					mm = attention_i[i][j][1:-1,1:-1].max()
					if mm > 0.5:
						indi = 0
						indj = 0
						for k in np.argmax(attention_i[i][j][1:-1,1:-1],axis=0):
							if mm in attention_i[i][j][1:-1,1:-1][k]:
								indi = k
								indj = np.argmax(attention_i[i][j][1:-1,1:-1][k])
						if abs(indi-indj) != 0:
							att_heads.append(attention_i[i][j])
				if heads == 'Mean' and len(att_heads) > 0:
						importance_attention_matrices.append(np.array(att_heads).mean(axis=0))
				elif heads == 'Sum' and len(att_heads) > 0:
						importance_attention_matrices.append(np.array(att_heads).sum(axis=0))
			importance_attention_matrices = np.array(importance_attention_matrices)

			if layers == 'Mean':
				attention = importance_attention_matrices.mean(axis=0)
			elif layers == 'Sum':
				attention = importance_attention_matrices.sum(axis=0)
			elif layers == 'Multi':
				attention = importance_attention_matrices[0]
				for i in range(1, len(importance_attention_matrices)):
					attention = np.matmul(attention, importance_attention_matrices[i])
			attention = attention[0]
		interpretations = []
		for label in range(len(self.label_names)):
			interpretations.append(attention[1:-1])
		if self.sentence_level:
			interpretations = self.convert_to_sentence(tokens, interpretations)
		return interpretations