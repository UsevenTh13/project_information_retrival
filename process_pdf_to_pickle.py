import fitz
import re
import string
import pickle

def extract_thesaurus(pdf_path):
    print(f"Membaca PDF dari {pdf_path} dengan ekstraksi pola Bold...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error membuka file: {e}")
        return []
        
    sentences = []
    curr_main = ''
    curr_syns = ''
    
    # Memproses seluruh halaman
    for i, page in enumerate(doc):
        # Beberapa halaman awal biasanya adalah daftar isi atau pengantar.
        for block in page.get_text('dict').get('blocks', []):
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text']
                        if 'Bold' in span['font']:
                            if curr_syns.strip():
                                process_entry(curr_main, curr_syns, sentences)
                                curr_main = ''
                                curr_syns = ''
                            curr_main += text
                        else:
                            curr_syns += text
                            
    if curr_syns.strip():
        process_entry(curr_main, curr_syns, sentences)
        
    return sentences

def process_entry(main_word, synonyms_text, sentences_list):
    main_word = main_word.strip().lower()
    main_word = re.sub(r'\d+', '', main_word).strip()
    
    if len(main_word) < 2 or not main_word.isalpha():
        return
        
    synonyms_text = synonyms_text.lower()
    synonyms_text = re.sub(r'\d+', '', synonyms_text)
    synonyms_text = synonyms_text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = synonyms_text.split()
    
    stop_words = {"yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah", "ini", "itu", "atau", "juga", "jadi"}
    pos_tags = {"n", "v", "a", "cak", "ki", "ant"}
    
    filtered_tokens = [w for w in tokens if w not in stop_words and w not in pos_tags and len(w) > 2]
    
    if filtered_tokens:
        sentence = [main_word] + filtered_tokens
        sentences_list.append(sentence)

if __name__ == '__main__':
    pdf_path = './KamusThesaurus.pdf'
    pickle_path = './thesaurus_dict_tokens.pkl'
    
    sentences = extract_thesaurus(pdf_path)
    print(f"Berhasil mengekstrak {len(sentences)} baris pola kamus yang valid.")
    
    print(f"Menyimpan hasil ke {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(sentences, f)
        
    print("Selesai! Pola kamus sudah sempurna.")
