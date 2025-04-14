import random
from collections import defaultdict
from nltk import trigrams
from nltk.tokenize import sent_tokenize, word_tokenize


class TrigramModel:
    def __init__(self, corpus_path, laplace_smoothing=False, alpha=1):
        self.corpus_path = corpus_path
        self.laplace_smoothing = laplace_smoothing
        self.alpha = alpha  # Smoothing parameter
        self.model = self.build_trigram_model()

    def load_corpus(self):
        encoding = 'utf-8'
        if self.corpus_path.endswith('.csv'):
            encoding = 'ISO-8859-1'
            
        with open(self.corpus_path, 'r', encoding=encoding) as f:
            raw_text = f.read().replace('[', '').replace(']', '').strip()
        return raw_text

    def tokenize_text(self, text):
        sentences = [word_tokenize(sent) for sent in sent_tokenize(text)]
        return [['<s>'] + sentence + ['</s>'] for sentence in sentences]

    def build_trigram_model(self):
        raw_text = self.load_corpus()
        tokenized_sentences = self.tokenize_text(raw_text)
        words = [word for sentence in tokenized_sentences for word in sentence]
        tri_grams = list(trigrams(words))

        trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Count actual trigrams in the corpus
        for w1, w2, w3 in tri_grams:
            trigram_counts[w1][w2][w3] += 1

        # If Laplace smoothing is enabled, add all possible trigrams (including unseen)
        if self.laplace_smoothing:
            vocab = list(set(words))  # Vocabulary
            vocab_size = len(vocab)  # Vocabulary size
            trigram_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

            # Add all possible trigrams to the model for smoothing (even unseen ones)
            for w1 in trigram_counts:
                for w2 in trigram_counts[w1]:
                    total_count = sum(trigram_counts[w1][w2].values()) + self.alpha * vocab_size  # Total with smoothing
                    
                    # Add smoothing for unseen trigrams
                    for w3 in vocab:  # All possible words in the vocabulary
                        count = trigram_counts[w1][w2].get(w3, 0)
                        trigram_probs[w1][w2][w3] = (count + self.alpha) / total_count
                        
            return trigram_probs
        else:
            # No smoothing: Use raw counts (ignore unseen trigrams)
            return trigram_counts

    def predict_next_word(self, w1, w2):
        if w1 in self.model and w2 in self.model[w1]:
            filtered_probs = self.model[w1][w2]
            words, probabilities = zip(*filtered_probs.items())
            return random.choices(words, weights=probabilities, k=1)[0]
        return "NONE"

    def generate_sentence(self, start_word, max_length=30):
        if start_word == "":
            start_word = random.choice(list(self.model['<s>'].keys()))

        string = ['<s>', start_word]
        sentence = [start_word]

        for _ in range(max_length):
            next_word = self.predict_next_word(string[-2], string[-1])
            if next_word == '</s>':
                break
            if next_word in {'.', ',', '!', '?', ';', ':', '"', "'"}:  # Attach punctuation directly
                sentence[-1] += next_word
            else:
                sentence.append(next_word)
            string.append(next_word)

        return ' '.join(sentence).replace('</s>', '').replace('NONE', '').strip()
