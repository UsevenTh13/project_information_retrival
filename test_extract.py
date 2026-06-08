import fitz
import re
import string

def extract_thesaurus(pdf_path):
    doc = fitz.open(pdf_path)
    sentences = []
    
    curr_main = ''
    curr_syns = ''
    
    for page in doc[100:105]: # Test on a few pages
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
    sentences = extract_thesaurus('KamusThesaurus.pdf')
    print("Total Extracted Entries:", len(sentences))
    for s in sentences[:10]:
        print(s[0], "->", s[1:5], "...")
