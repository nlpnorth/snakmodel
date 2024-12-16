import os

txtFiles = ['data/twitter/twitter.da.all.txt', 'data/ftlm/ft_lm_train_data.txt', 'data/reddit.da/utterances.txt', 'data/bookshop/bookshop.da.txt', 'data/opensubtitles/opensubtitle.da.txt', 'data/dawiki/dawiki-latest-pages-articles.txt', 'data/cc100/cc-100-da.txt', 'data/culturax/culturaX.txt' , 'data/gigaword/all.txt', 'data/danewsroom/danewsroom.txt']
txtFiles = ['_raw_data/medical/hospital.txt']

class ScriptFinder():
    def __init__(self):
        """
        Class that loads the scripts definitions from Unicode; it automatically
        downloads them to a text file, and saves them in an ordered list of lists
        (triples). Where each script is represented as a beginning index, end 
        index and name.
        """
        self.ranges = [None] * 918000
        if not os.path.isfile('scripts/Scripts.txt'):
            os.system('mkdir -p scripts')
            os.system('wget https://www.unicode.org/Public/15.0.0/ucd/Scripts.txt --no-check-certificate -O scripts/Scripts.txt')
        for line in open('scripts/Scripts.txt'):
            tok = line.split(';')
            if line[0]!='#' and len(tok) == 2:
                char_range_hex = tok[0].strip().split('..')
                char_range_int = [int(x, 16) for x in char_range_hex]
                script_name = tok[1].strip().split()[0]
                if len(char_range_int) == 1:
                    self.ranges[char_range_int[0]] = script_name
                else:
                    for ind in range(char_range_int[0], char_range_int[1]+1):
                        self.ranges[ind] = script_name
                # Note that we include the first and the last character of the
                # range in the indices, so the first range for Latin is 65-90
                # for example, character 65 (A) and 90 (Z) are both included in
                # the Latin set.  


    def find_char(self, char):
        """
        Return the script of a single character, if a string
        is passed, it returns the script of the first character.

        Parameters
        ----------
        char: char
            The character to find the script of, if this is a string
            the first character is used.
    
        Returns
        -------
        script: str
            The name of the script, or None if not found
        """
        if len(char) > 1:
            char = char[0]
        char_idx = ord(char)
        if char_idx >= len(self.ranges):
            return None
        return self.ranges[char_idx]

