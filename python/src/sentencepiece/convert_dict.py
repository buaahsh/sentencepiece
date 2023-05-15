with open("dict.250k.txt", 'r') as f1, open('dict.txt', 'w', encoding='utf-8') as f2:
    for line in f1:
        token = line.split('\t')[0]
        f2.write(f'{token} 1\n')