import json

def main():
    with open('qa_Video_Games.json', 'r') as f_in, open('corpus.txt', 'w') as f_out:
        for line in f_in:
            data = eval(line)
            f_out.write(data['question'] + '\n')
            f_out.write(data['answer'] + '\n')

if __name__ == '__main__':
    main()
