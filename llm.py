from openai import OpenAI
import misc
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
        

    def get_response(self, pred, system_content=None, retry_amt=3):
        if system_content is None:
            system_content = self._get_default_system_content()
        
        user_content = self._lines2str(pred)
        response = self._get_response(user_content, system_content)
        all_valid_lines = self._validate_lines(response)
        response[~all_valid_lines] = pred[~all_valid_lines]

        all_response = response
        while not np.all(all_valid_lines) and retry_amt > 0:
            user_content = self._lines2str(all_response[~all_valid_lines])
            response = self._get_response(user_content, system_content)
            all_response[~all_valid_lines] = response
            all_valid_lines = self._validate_lines(all_response)
            all_response[~all_valid_lines] = pred[~all_valid_lines]
            retry_amt -= 1

        return all_response
    

    def _get_response(self, user_content, system_content):
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        )

        response = completion.choices[0].message.content
        response = self._format_response(response)
        return response
    
    def _get_default_system_content(self):
        str_corpus = ', '.join(self.corpus)
        str_corpus = '[' + str_corpus + ']'
        examples = ['bin blue at f one now', 'set red with v two again', 'place green by d one please']
        str_examples = ', '.join(examples)
        system_content = "You are tasked with final sentence inference. Respond only with exactly a sequence of 6 words or characters for every line from the following corpus:" + str_corpus + " here are several examples: " + str_examples + ". Do not respond with any other words or characters. Do not respond with more lines than the number of lines in the prompt."
        return system_content
    
    def _format_response(self, response):
        response = re.sub(r'[0-9]. ', '', response)
        response = re.sub(r'[^\w\s]', '', response)
        response = response.lower()
        lines = response.split('\n')
        lines = [line.strip() for line in lines]
        lines = [line.split(' ') for line in lines]
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
    api_key = "your-api-key"
    send_api = True

    llm = LLM_Inference(api_key)
    pred = np.array([['place', 'at', 'd', 'four', 'again', 'please'],['bin', 'blue', 'at', 'z', 'three', 'please']])
    response = llm.get_response(pred)
    print(response)