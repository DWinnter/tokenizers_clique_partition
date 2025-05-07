from collections import defaultdict
import re
from corpus_math import corpus
from test_sentence import test_sentence
from params import aim_token_num
from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count
from multiprocessing import freeze_support
import json
from collections import OrderedDict
import tiktoken
from collections import Counter


    
def calculate_ngrams(args):
    n, vocab_chunk = args
    result = {}
    for word, freq in vocab_chunk.items():
        symbols = word.split()
        for i in range(len(symbols) - n + 1):
            ngram = tuple(symbols[i:i+n])
            if ngram not in result:
                result[ngram] = {'freq': 0, 'score': 0}
            result[ngram]['freq'] += freq
            result[ngram]['score'] += n * freq
    return result

def process_n_value(args):
    n, vocab_chunks = args
    num_processes = min(cpu_count(), len(vocab_chunks))
    with Pool(processes=num_processes) as pool_v:
        results_v = pool_v.map(calculate_ngrams, [(n, chunk) for chunk in vocab_chunks])

    combined_stats_n = {}
    for result in results_v:
        for ngram, data in result.items():
            if ngram in combined_stats_n:
                combined_stats_n[ngram]['freq'] += data['freq']
                combined_stats_n[ngram]['score'] += data['score']
            else:
                combined_stats_n[ngram] = data
    return combined_stats_n

class LengthTokenizer:
    def __init__(self, corpus, num_merges):
        self.corpus = corpus
        self.num_merges = num_merges
        self.vocab = defaultdict(int)
        self.merges = {}
        self.train()

    def get_vocab(self):
        for sentence in self.corpus:
            sequence = ' '.join(' '.join(list(word) + ['</w>']) for word in sentence.split())
            self.vocab[sequence] += 1

    def get_stats(self):
        vocab_items = list(self.vocab.items())
        num_processes = cpu_count()
        chunk_size = len(vocab_items) // num_processes + 1
        vocab_chunks = [dict(vocab_items[i:i+chunk_size]) for i in range(0, len(vocab_items), chunk_size)]
        n_values = [2, 3, 4, 5]

        tasks = [(n, chunk) for n in n_values for chunk in vocab_chunks]

        with Pool(processes=num_processes) as pool:
            results = pool.map(calculate_ngrams, tasks)

        combined_stats = {}
        for result in results:
            for ngram, data in result.items():
                if ngram in combined_stats:
                    combined_stats[ngram]['freq'] += data['freq']
                    combined_stats[ngram]['score'] += data['score']
                else:
                    combined_stats[ngram] = data

        return combined_stats

    def train(self):
        self.get_vocab()
        self.print_vocab()
        
        for i in range(self.num_merges):
            stats = self.get_stats()
            if not stats:
                break
            
            best = None
            best_freq = 0
            best_score = 0
            
            while stats:
                current_best = max(stats, key=lambda x: stats[x]['score'])
                current_freq = stats[current_best]['freq']
                current_score = stats[current_best]['score']
                
                if current_freq > 1:
                    best = current_best
                    best_freq = current_freq
                    best_score = current_score
                    break
                else:
                    del stats[current_best]
            
            if not best:
                print("没有更多频率>=1的组合可供选择，停止训练")
                break
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"当前最佳组合，第{self.get_token_count()}个：{best}，频率：{best_freq}，得分：{best_score}")

            if self.get_token_count() >= aim_token_num:
                break
            
            self.merge_vocab(best)
            self.print_vocab()

        print("\n训练完成，最终的token列表:")
        self.print_final_tokens()

    def save_token_table(self):
        token_table = {
            'merges': OrderedDict((' '.join(k), v) for k, v in self.merges.items()),
            'vocab': dict(self.vocab)
        }
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        token_table_filename = f"token_table_{timestamp}.json"
        with open(token_table_filename, 'w', encoding='utf-8') as f:
            json.dump(token_table, f, ensure_ascii=False, indent=2)
        print(f"Token表已保存到 {token_table_filename}")
        return token_table_filename
    
    @classmethod
    def load_from_token_table(cls, token_table_path):
        with open(token_table_path, 'r', encoding='utf-8') as f:
            token_table = json.load(f)
        
        tokenizer = cls.__new__(cls)

        def effective_length(token):
            return len(token.replace('</w>', '-'))

        sorted_merges = sorted(token_table['merges'].items(), key=lambda x: effective_length(x[1]), reverse=True)




        tokenizer.merges = OrderedDict((tuple(k.split()), v) for k, v in token_table['merges'].items())
        tokenizer.vocab = defaultdict(int, token_table['vocab'])
        return tokenizer

    def merge_vocab(self, best):
        ngram = re.escape(' '.join(best))
        replacement = ''.join(best)
        
        pattern = re.compile(r'(?<!\S)' + ngram + r'(?!\S)')
        
        self.merges[best] = replacement
        
        new_vocab = {}
        for word, freq in self.vocab.items():
            new_word = pattern.sub(replacement, word)
            new_vocab[new_word] = freq
        self.vocab = new_vocab

    def print_vocab(self):
        return
        # for word, freq in self.vocab.items():
        #     logging.info(f"  {word}: {freq}")

    def get_token_count(self):
        tokens = set()
        for word in self.vocab.keys():
            tokens.update(word.split())
        return len(tokens)

    def print_final_tokens(self):
        tokens = set()
        for word in self.vocab.keys():
            tokens.update(word.split())
        tokens = sorted(list(tokens))
        print("  " + ", ".join(tokens))
        print(f"  总token数: {len(tokens)}")
        
    def tokenize(self, text):
        tokens = []
        
        if isinstance(text, list):
            for sentence in text:
                tokens.extend(self.tokenize_sentence(sentence))
        else:
            tokens = self.tokenize_sentence(text)
        
        return tokens

    def tokenize_sentence(self, sentence):
        chars = []
        for word in sentence.split():
            chars.extend(list(word) + ['</w>'])


        def effective_length(token):
            return len(token.replace('</w>', '-'))
        sorted_merges = sorted(self.merges.items(), key=lambda x: effective_length(x[1]), reverse=True)


        for pair, merge in sorted_merges:
            window_size = effective_length(merge)
            #window_size = len(merge)
            i = 0
            for j in range(i, len(chars) - window_size + 1):
                window = ''.join(chars[j:j+window_size])
                if window == merge:
                    chars[j] = merge
                    del chars[j+1:j+window_size]

        return chars

    def effective_length(self, token):
        return len(token.replace('</w>', '-'))

    def calculate_average_token_length(self):
        total_length = sum(len(token) for token in self.get_all_tokens())
        token_count = self.get_token_count()
        return total_length / token_count if token_count > 0 else 0

    def get_all_tokens(self):
        tokens = set()
        for word in self.vocab.keys():
            tokens.update(word.split())
        return tokens

if __name__ == '__main__':
    freeze_support()
    
    log_filename = f"tokenizer_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8')

    tokenizer = LengthTokenizer(corpus, num_merges=20000)
    token_table_filename = tokenizer.save_token_table()

    loaded_tokenizer = LengthTokenizer.load_from_token_table(token_table_filename)
    
    tokens = loaded_tokenizer.tokenize(test_sentence)
   
    logging.info("Token种数: %d", loaded_tokenizer.get_token_count())
    logging.info("Token个数: %d", len(tokens))

    original_word_count = len(test_sentence.split())
    logging.info("原始单词数量: %d", original_word_count)
    logging.info("Token数量与原始单词数量的比率: %.2f", len(tokens) / original_word_count)

    total_chars = sum(len(sentence) for sentence in test_sentence)
    avg_token_length = total_chars / len(tokens) if tokens else 0
    logging.info("Token的平均长度: %.2f", avg_token_length)
    logging.info("Tokenized: %s", tokens)
