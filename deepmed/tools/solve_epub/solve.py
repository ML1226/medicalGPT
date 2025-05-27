if __name__ == '__main__':
    fi = open("text00002.html")
    fo = open("text00002.txt", "w")
    flag = 0
    for line in fi:
        if flag == 0:
            if not line.startswith('<p class="calibre2">'):
                continue
            if line.endswith('</p>\n'):
                fo.write(line[20:-5] + '\n')
            else:
                fo.write(line[20:-1])
                flag = 1
        elif flag == 1:
            if line.endswith('</p>\n'):
                fo.write(line[:-5] + '\n')
                flag = 0
            else:
                fo.write(line[:-1])
    fi.close()
    fo.close()
        