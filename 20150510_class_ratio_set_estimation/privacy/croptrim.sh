pdftk A=$1 cat A1 output t.pdf
pdfcrop t.pdf t1.pdf
mv t1.pdf $1
rm t.pdf
