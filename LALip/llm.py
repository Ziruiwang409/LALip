from openai import OpenAI
import LALip.misc as misc
# import misc as misc
import time
import re
import numpy as np

class LLM_Inference:

    def __init__(self,
                 api_key,
                 model="gpt-4"):
        
        self.api_key = api_key
        self.model = model
        self.corpus = misc.gt_label()
        self.client = OpenAI(api_key=api_key)
        

    def get_response(self, pred, prompt=None, retry_amt=3):
        if prompt is None:
            prompt = self._get_prompt()
        
        user_content = self._lines2str(pred)
        response = self._get_response(user_content, prompt)

        # if len(response) == len(pred):
        #     all_valid_lines = self._validate_lines(response)
        #     response[~all_valid_lines] = pred[~all_valid_lines]
        # else:
        #     print("response length does not match prediction length")
        #     all_valid_lines = np.array([False] * len(pred))
        #     response = pred

        all_response = response
        # while not np.all(all_valid_lines) and retry_amt > 0:
        #     user_content = self._lines2str(all_response[~all_valid_lines])
        #     response = self._get_response(user_content, prompt)
        #     if len(response) == len(all_response[~all_valid_lines]):
        #         all_response[~all_valid_lines] = response
        #         all_valid_lines = self._validate_lines(all_response)
        #         all_response[~all_valid_lines] = pred[~all_valid_lines]
            
        #     retry_amt -= 1


        return all_response
    

    def _get_response(self, user_content, prompt):
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content}
        ]
        )

        response = completion.choices[0].message.content
        response = self._format_response(response)

        return response
    
    def _get_prompt(self):
        str_corpus = ', '.join(self.corpus)
        str_corpus = '[' + str_corpus + ']'
        prompt = '''You are a agent whose job is to do text correction. 
                    In this task, you are given a list of exact 6 tokens, where each token can be a character or string.
                    Your job is to correct the tokens to match the following rules:     
                        1. After correction, each token should be in the following list: ''' + str_corpus + '''
                        2. The number of tokens should be the same as the original list
                        3. The order of tokens should be the same as the original list
                        4. If the original token is in the corpus, the corrected token should be the same as the original token
                        5. If the original token is not in the corpus, the corrected token should be a element from the list I provided you
                        6. If you are not sure about the correction, you can leave it as it is
                        7. Return value should be the concatenation string of the corrected tokens, separated by space
                    
                    Here are some examples to help you understand the rules:
                        [bin, blue, at, f, one, now] -> bin blue at f one now
                        ['place' 'at' 'd' 'four' 'again' 'please'] -> place at d four again please
                        ['plac' 'whited' 'wit' 'n' 'nine' 'now'] -> place white with n nine now
                        ['place' 'white' 'by' 'z' 'zwuo' 'now'] -> place white by z two now  
                        ...
                    and output should be:
                        bin blue at f one now
                        place at d four again please
                        place white with n nine now
                        place white by z two now
                        ...
                    Based on the above rules, please correct the following tokens:
                    '''
        return prompt
    
    def _format_response(self, response):
        lines = response.split('\n')
        # remove the first token for each line and concat back to string
        lines = [str(line.split(' ')[1:]) for line in lines]
        return np.array(lines)
    
    def _validate_lines(self, lines):
        def _validate_length(line):
            if len(line) != 6:
                print("not 6 words: " + line)
                return False
            return True
        
        def _validate_words(line):
            for word in line:
                if word not in self.corpus:
                    print("not in corpus: " + word)
                    return False
            return True
        
        lines = [line.strip() for line in lines]
        lines = [line.split(' ') for line in lines]

        valid_lst = [True] * len(lines)
        validation_fns = [_validate_length, _validate_words]

        for (i, line) in enumerate(lines):
            for fn in validation_fns:
                valid_lst[i] = valid_lst[i] and fn(line)

        return np.array(valid_lst)
    
    def _lines2str(self, lines):
        ret_str = ''
        for i in range(len(lines)):
            ret_str += str(i+1) + '. ' + ' '.join(lines[i]) + '\n '

        return ret_str        

if __name__ == "__main__":
    api_key = "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"

    llm = LLM_Inference(api_key)
    pred = np.array([['lay','gren', 'it', 'u', 'four', 'soon'], 
                     ['set blue it z seven soon'], 
                     ['lay white in p five now'], 
                     ['bin white in c nine please'], 
                     ['lay white with z four again']]) 
    response = llm.get_response(pred)
    print(type(response))
    print(type(response[0]))
    print(response[0])