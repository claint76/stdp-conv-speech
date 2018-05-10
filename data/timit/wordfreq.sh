find . -name "*.wrd"|xargs cat|cut -d ' ' -f 3|sort|uniq -c|sort -n -r
